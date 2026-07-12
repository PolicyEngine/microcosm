"""Pinned enhanced-FRS input-surface reference for UK coverage parity.

The licensed H5 never ships in Populace.  ``efrs_parity_reference.json`` is the
frozen, sha-addressed extraction of its populated PolicyEngine-UK input leaves.
Release coverage manifests and gates consume this small checked-in fact instead
of resolving a moving or licensed artifact at build time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

__all__ = [
    "EFRS_PARITY_KNOWN_GAPS_RESOURCE",
    "EFRS_PARITY_REFERENCE_RESOURCE",
    "EfrsParityKnownGap",
    "EfrsParityReference",
    "EfrsParitySource",
    "load_efrs_parity_known_gaps",
    "load_efrs_parity_reference",
]

EFRS_PARITY_REFERENCE_RESOURCE = "efrs_parity_reference.json"
EFRS_PARITY_KNOWN_GAPS_RESOURCE = "efrs_parity_known_gaps.json"

_UK_PACKAGE = "populace.build.uk"


@dataclass(frozen=True)
class EfrsParitySource:
    """Immutable coordinates and byte identity of the enhanced-FRS artifact."""

    repo_id: str
    repo_type: str
    filename: str
    revision: str
    sha256: str
    url: str
    vintage: str
    period: str
    size_bytes: int


@dataclass(frozen=True)
class EfrsParityReference:
    """Frozen per-input populated shares plus source provenance."""

    source: EfrsParitySource
    nonzero_shares: Mapping[str, float]
    input_entities: Mapping[str, str]
    schema_version: int = 3

    @property
    def populated_layers(self) -> tuple[str, ...]:
        return tuple(name for name, share in self.nonzero_shares.items() if share > 0.0)


@dataclass(frozen=True)
class EfrsParityKnownGap:
    """One honestly reasoned, ledger-tracked reference coverage gap."""

    variable: str
    reason: str
    tracking_note: str


def _resource_text(resource: str) -> str:
    candidate = Path(resource)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return files(_UK_PACKAGE).joinpath(resource).read_text(encoding="utf-8")


def _resource_payload(resource: str) -> Mapping[str, object]:
    payload = json.loads(_resource_text(resource))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resource}: expected a JSON object.")
    return payload


def _require_str(value: object, *, field_name: str, resource: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{resource}: field {field_name!r} must be a non-empty string, "
            f"got {value!r}."
        )
    return value


def load_efrs_parity_reference(
    resource: str = EFRS_PARITY_REFERENCE_RESOURCE,
) -> EfrsParityReference:
    """Load and validate the frozen enhanced-FRS populated-input reference."""
    payload = _resource_payload(resource)
    raw_source = payload.get("source")
    if not isinstance(raw_source, Mapping):
        raise ValueError(f"{resource}: 'source' must be a JSON object.")

    size_bytes = raw_source.get("size_bytes")
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError(f"{resource}: 'source.size_bytes' must be positive.")
    source = EfrsParitySource(
        repo_id=_require_str(
            raw_source.get("repo_id"),
            field_name="source.repo_id",
            resource=resource,
        ),
        repo_type=_require_str(
            raw_source.get("repo_type"),
            field_name="source.repo_type",
            resource=resource,
        ),
        filename=_require_str(
            raw_source.get("filename"),
            field_name="source.filename",
            resource=resource,
        ),
        revision=_require_str(
            raw_source.get("revision"),
            field_name="source.revision",
            resource=resource,
        ),
        sha256=_require_str(
            raw_source.get("sha256"),
            field_name="source.sha256",
            resource=resource,
        ),
        url=_require_str(
            raw_source.get("url"), field_name="source.url", resource=resource
        ),
        vintage=_require_str(
            raw_source.get("vintage"),
            field_name="source.vintage",
            resource=resource,
        ),
        period=_require_str(
            raw_source.get("period"),
            field_name="source.period",
            resource=resource,
        ),
        size_bytes=size_bytes,
    )

    raw_shares = payload.get("nonzero_shares")
    if not isinstance(raw_shares, Mapping) or not raw_shares:
        raise ValueError(
            f"{resource}: 'nonzero_shares' must be a non-empty JSON object; a "
            "silently-empty reference would make the coverage gate vacuous."
        )
    shares: dict[str, float] = {}
    for name, value in raw_shares.items():
        try:
            share = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{resource}: share for {name!r} is not a number ({value!r})."
            ) from exc
        if not math.isfinite(share) or not (0.0 <= share <= 1.0):
            raise ValueError(
                f"{resource}: share for {name!r} is {share!r}, outside [0, 1]."
            )
        shares[str(name)] = share

    raw_entities = payload.get("input_entities")
    if not isinstance(raw_entities, Mapping):
        raise ValueError(f"{resource}: 'input_entities' must be a JSON object.")
    entities = {
        str(name): _require_str(
            entity,
            field_name=f"input_entities[{name}]",
            resource=resource,
        )
        for name, entity in raw_entities.items()
    }
    expected_entities = {"person", "benunit", "household"}
    invalid_entities = {
        name: entity for name, entity in entities.items() if entity not in expected_entities
    }
    if invalid_entities:
        raise ValueError(
            f"{resource}: input entity values must be one of "
            f"{sorted(expected_entities)}, got {invalid_entities}."
        )
    missing_entities = sorted(set(shares) - set(entities))
    extra_entities = sorted(set(entities) - set(shares))
    if missing_entities or extra_entities:
        raise ValueError(
            f"{resource}: input_entities must exactly cover nonzero_shares; "
            f"missing={missing_entities}, extra={extra_entities}."
        )

    schema_version = payload.get("schema_version", 1)
    if not isinstance(schema_version, int):
        raise ValueError(f"{resource}: 'schema_version' must be an integer.")
    return EfrsParityReference(
        source=source,
        nonzero_shares=shares,
        input_entities=entities,
        schema_version=schema_version,
    )


def load_efrs_parity_known_gaps(
    resource: str = EFRS_PARITY_KNOWN_GAPS_RESOURCE,
) -> tuple[EfrsParityKnownGap, ...]:
    """Load the canonical UK exclusion ledger (which may honestly be empty)."""
    payload = _resource_payload(resource)
    raw_gaps = payload.get("known_gaps")
    if not isinstance(raw_gaps, Mapping):
        raise ValueError(f"{resource}: 'known_gaps' must be a JSON object.")
    gaps: list[EfrsParityKnownGap] = []
    for variable, entry in sorted(raw_gaps.items()):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{resource}: entry for {variable!r} must be a JSON object."
            )
        gaps.append(
            EfrsParityKnownGap(
                variable=str(variable),
                reason=_require_str(
                    entry.get("reason"),
                    field_name=f"known_gaps[{variable}].reason",
                    resource=resource,
                ),
                tracking_note=_require_str(
                    entry.get("tracking_note"),
                    field_name=f"known_gaps[{variable}].tracking_note",
                    resource=resource,
                ),
            )
        )
    return tuple(gaps)
