"""Pinned incumbent-eCPS parity reference and the reason'd exemption register.

The launch contract for replacing the enhanced-CPS (:func:`microcosm.build.gates
.parity_gate`) is share-based: every layer the incumbent populates, the
candidate populates, with exemptions only by documented name. That contract
needs two checked-in facts, both owned here:

- :func:`load_ecps_parity_reference` — the frozen per-variable nonzero record
  shares of the incumbent artifact (``ecps_parity_reference.json``), computed
  ONCE from the pinned, sha-verified ``policyengine-us-data`` enhanced-CPS
  release. It is never recomputed from a moving artifact; the release compares
  a live candidate against this file. The source artifact's sha256 and vintage
  travel with the shares so a release manifest can state exactly which
  incumbent it measured parity against. Reviewed consumerless legacy inputs
  may be retired from the frozen requirement surface only when named in the
  reference's ``reviewed_consumerless_inputs_excluded`` receipt.
- :func:`load_ecps_parity_known_gaps` — the exemption register
  (``ecps_parity_known_gaps.json``): every layer the incumbent populates that
  the current candidate does not, each carrying a reason and a tracking issue.
  This is the debt ledger the parity gate's ``known_gaps`` is drawn from; the
  launch condition is that it shrinks to deliberate scope decisions, not TODOs.

Both files are declared resources of the US country package
(``country_package.json``) and are read with :mod:`importlib.resources`, the
same mechanism the fiscal-target references use.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

__all__ = [
    "ECPS_PARITY_REFERENCE_RESOURCE",
    "ECPS_PARITY_KNOWN_GAPS_RESOURCE",
    "EcpsParityReference",
    "EcpsParitySource",
    "ParityKnownGap",
    "load_ecps_parity_reference",
    "load_ecps_parity_known_gaps",
]

ECPS_PARITY_REFERENCE_RESOURCE = "ecps_parity_reference.json"
ECPS_PARITY_KNOWN_GAPS_RESOURCE = "ecps_parity_known_gaps.json"

_US_PACKAGE = "microcosm.build.us"


@dataclass(frozen=True)
class EcpsParitySource:
    """Provenance of the pinned incumbent artifact the shares were computed from.

    Attributes:
        repo_id: Hugging Face repo id (``policyengine/policyengine-us-data``).
        repo_type: ``"model"`` or ``"dataset"`` — the incumbent eCPS ships in
            the model repo.
        filename: Artifact filename (``enhanced_cps_2024.h5``).
        revision: The repo revision pinned at computation time.
        sha256: The artifact file's content sha256 (== its Hugging Face LFS
            object id), so a consumer can prove it measured the same bytes.
        vintage: The data vintage (e.g. ``"2024"``).
        period: The period the shares were read at (e.g. ``"2024"``).
    """

    repo_id: str
    repo_type: str
    filename: str
    revision: str
    sha256: str
    vintage: str
    period: str


@dataclass(frozen=True)
class EcpsParityReference:
    """The frozen parity reference: nonzero shares plus artifact provenance.

    Attributes:
        source: Provenance of the incumbent artifact.
        nonzero_shares: Variable -> share of records with a non-zero (or True)
            value in the incumbent artifact, over the engine's input-variable
            surface (formula-owned and structural id columns excluded). Only
            populated layers (share > 0) are stored; an unpopulated incumbent
            layer is not a parity requirement.
        schema_version: The reference file schema version.
    """

    source: EcpsParitySource
    nonzero_shares: Mapping[str, float]
    schema_version: int = 1

    @property
    def populated_layers(self) -> tuple[str, ...]:
        """Every reference layer the candidate must populate or exempt."""
        return tuple(name for name, share in self.nonzero_shares.items() if share > 0.0)


@dataclass(frozen=True)
class ParityKnownGap:
    """One exemption-register entry: a named gap with a reason and an issue.

    Attributes:
        variable: The exempted variable name (a reference-populated layer the
            current candidate does not populate).
        reason: Why the layer is absent — a documented decision, never empty.
        issue: The tracking issue that owns closing the gap (e.g.
            ``"PolicyEngine/microcosm#312"``).
    """

    variable: str
    reason: str
    issue: str


def _resource_text(resource: str) -> str:
    """Read a US-package resource by name, or a filesystem path if one is given.

    The default callers pass a packaged resource filename read via
    :mod:`importlib.resources`; tests (and any out-of-tree override) may pass a
    path to an alternate JSON file.
    """
    candidate = Path(resource)
    if candidate.exists():
        return candidate.read_text()
    return files(_US_PACKAGE).joinpath(resource).read_text()


def _resource_payload(resource: str) -> Mapping[str, object]:
    raw = json.loads(_resource_text(resource))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{resource}: expected a JSON object.")
    return raw


def _require_str(value: object, *, field_name: str, resource: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{resource}: field {field_name!r} must be a non-empty string, "
            f"got {value!r}."
        )
    return value


def load_ecps_parity_reference(
    resource: str = ECPS_PARITY_REFERENCE_RESOURCE,
) -> EcpsParityReference:
    """Load the pinned incumbent-eCPS parity reference from the US package.

    Raises:
        ValueError: If the source provenance is incomplete, a share is not a
            finite number in ``[0, 1]``, or the payload shape is wrong. A
            malformed reference must fail loudly — a silently-empty reference
            would make the parity gate vacuously pass.
    """
    payload = _resource_payload(resource)
    raw_source = payload.get("source")
    if not isinstance(raw_source, Mapping):
        raise ValueError(f"{resource}: 'source' must be a JSON object.")
    source = EcpsParitySource(
        repo_id=_require_str(
            raw_source.get("repo_id"), field_name="source.repo_id", resource=resource
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
            raw_source.get("sha256"), field_name="source.sha256", resource=resource
        ),
        vintage=_require_str(
            raw_source.get("vintage"),
            field_name="source.vintage",
            resource=resource,
        ),
        period=_require_str(
            raw_source.get("period"), field_name="source.period", resource=resource
        ),
    )

    raw_shares = payload.get("nonzero_shares")
    if not isinstance(raw_shares, Mapping) or not raw_shares:
        raise ValueError(
            f"{resource}: 'nonzero_shares' must be a non-empty JSON object; a "
            "silently-empty reference would make the parity gate vacuous."
        )
    shares: dict[str, float] = {}
    for name, value in raw_shares.items():
        try:
            share = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{resource}: share for {name!r} is not a number ({value!r})."
            ) from exc
        if not (0.0 <= share <= 1.0):
            raise ValueError(
                f"{resource}: share for {name!r} is {share!r}, outside [0, 1]."
            )
        shares[str(name)] = share

    schema_version = payload.get("schema_version", 1)
    if not isinstance(schema_version, int):
        raise ValueError(f"{resource}: 'schema_version' must be an integer.")

    return EcpsParityReference(
        source=source,
        nonzero_shares=shares,
        schema_version=schema_version,
    )


def load_ecps_parity_known_gaps(
    resource: str = ECPS_PARITY_KNOWN_GAPS_RESOURCE,
) -> tuple[ParityKnownGap, ...]:
    """Load the reason'd, issue-linked parity exemption register.

    Returns the register entries in variable-name order.

    Raises:
        ValueError: If any entry is missing a reason or an issue reference —
            an undocumented exemption is a silent zero with extra steps.
    """
    payload = _resource_payload(resource)
    raw_gaps = payload.get("known_gaps")
    if not isinstance(raw_gaps, Mapping):
        raise ValueError(f"{resource}: 'known_gaps' must be a JSON object.")
    gaps: list[ParityKnownGap] = []
    for variable, entry in sorted(raw_gaps.items()):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{resource}: entry for {variable!r} must be a JSON object with "
                "'reason' and 'issue'."
            )
        reason = _require_str(
            entry.get("reason"),
            field_name=f"known_gaps[{variable}].reason",
            resource=resource,
        )
        issue = _require_str(
            entry.get("issue"),
            field_name=f"known_gaps[{variable}].issue",
            resource=resource,
        )
        gaps.append(ParityKnownGap(variable=str(variable), reason=reason, issue=issue))
    return tuple(gaps)
