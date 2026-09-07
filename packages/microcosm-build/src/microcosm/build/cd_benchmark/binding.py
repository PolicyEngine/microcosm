"""Candidate binding slots: caller evidence versus authenticated production.

``/candidate_binding_requirements`` lists twenty-one artifacts that must be
authenticated before scoring, and states the rule this module exists to keep:
"IDs and hashes must resolve to authenticated immutable artifacts bound to this
exact candidate/population, not prose labels", and "this contract supplies
required slots, not invented completed adapters or candidate identities".

So a filled slot is *caller evidence*. It is not authentication, and no number
of filled slots makes a mapping approved. Authentication is the job of a
producer authenticator satisfying :class:`ProducerAuthenticator`, of which this
package ships none: there is no default, no production instance and no
fallback. With no authenticator injected — the only state this implementation
slice can be in — every binding resolves to ``unapproved``, whatever it
contains. A fixture authenticator has to be written and passed in explicitly by
a test.

The document lists the slots but does not assign a reason code to each. The
partition below is this package's reading, chosen so a missing slot reports the
axis it actually belongs to rather than one catch-all code; the docs record it
as an implementation classification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from microcosm.build.cd_benchmark.canonical import is_sha256_hex
from microcosm.build.cd_benchmark.protocol import CompiledProtocol
from microcosm.build.cd_benchmark.reasons import MappingStatus, Reason, dedupe

__all__ = [
    "ArtifactRef",
    "BindingError",
    "BindingResult",
    "CandidateBinding",
    "EXPOSURE_SLOTS",
    "IDENTITY_SLOTS",
    "LINEAGE_SLOTS",
    "MAPPING_SLOTS",
    "ProducerAuthenticator",
    "resolve_binding",
]

#: Slots naming who produced this candidate and which population it is.
IDENTITY_SLOTS = (
    "candidate_id",
    "candidate_manifest_sha256",
    "population_artifact_id",
    "population_content_sha256",
    "population_semantic_snapshot_sha256",
    "producer_run_id",
    "producer_git_head_and_implementation_sha256",
    "model_identity_or_explicit_not_invoked",
)
#: Slots tracing every in-scope record to its original authenticated source.
LINEAGE_SLOTS = (
    "source_member_manifest_sha256",
    "source_lineage_mapping_artifact_id_and_sha256",
    "selection_clone_prune_and_export_ancestry_manifest_sha256",
)
#: Slots defining concept, period, universe, geography and reducer.
MAPPING_SLOTS = (
    "universe_definition_id_and_sha256_per_block_per_arm",
    "universe_classification_artifact_id_and_sha256",
    "period_mapping_definition_id_and_sha256",
    "income_eight_component_definition_id_and_sha256_when_income_claimed",
    "geography_assignment_artifact_id_and_sha256",
    "geography_definition_id_and_sha256",
    "weight_and_typed_reducer_definition_id_and_sha256",
)
#: Slots establishing reservation, tuning ancestry and ledger history.
EXPOSURE_SLOTS = (
    "tuning_ancestry_manifest_sha256",
    "reservation_audit_artifact_sha256",
    "exposure_ledger_snapshot_sha256",
)

_SLOT_REASONS = {
    **{name: Reason.CANDIDATE_IDENTITY_UNAVAILABLE for name in IDENTITY_SLOTS},
    **{name: Reason.UNRESOLVED_SOURCE_LINEAGE for name in LINEAGE_SLOTS},
    **{name: Reason.MAPPING_UNAPPROVED for name in MAPPING_SLOTS},
    **{name: Reason.EXPOSURE_HISTORY_UNKNOWN for name in EXPOSURE_SLOTS},
}


class BindingError(ValueError):
    """A binding slot is not a usable artifact reference."""


@dataclass(frozen=True)
class ArtifactRef:
    """An immutable artifact identifier and its content digest."""

    artifact_id: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise BindingError("An artifact reference needs a nonempty id.")
        if not is_sha256_hex(self.sha256):
            raise BindingError("An artifact reference needs a lowercase digest.")


@runtime_checkable
class ProducerAuthenticator(Protocol):
    """Resolves a slot's reference against the real immutable artifact.

    No implementation ships in this package. An implementation must actually
    open the named artifact, verify its digest, and verify that it is bound to
    this exact candidate and population — not merely echo the reference back.
    """

    def authenticates(self, slot: str, reference: ArtifactRef) -> bool:
        """Return whether ``reference`` resolves to an authenticated artifact."""


@dataclass(frozen=True)
class CandidateBinding:
    """The caller's declared slot references for one candidate evaluation."""

    slots: Mapping[str, ArtifactRef]
    income_claimed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.income_claimed, bool):
            raise BindingError("income_claimed must be a boolean.")
        slots = dict(self.slots)
        for name, value in slots.items():
            if not isinstance(value, ArtifactRef):
                raise BindingError(f"Slot {name!r} must hold an ArtifactRef.")
        object.__setattr__(self, "slots", MappingProxyType(slots))

    @classmethod
    def empty(cls) -> CandidateBinding:
        """A binding with nothing declared."""
        return cls(slots={})


@dataclass(frozen=True)
class BindingResult:
    """The mapping axis for one candidate binding."""

    status: MappingStatus
    missing_slots: tuple[str, ...]
    unauthenticated_slots: tuple[str, ...]
    unknown_slots: tuple[str, ...]
    reasons: tuple[Reason, ...]
    income_claimed: bool
    evidence: Mapping[str, object] = field(default_factory=dict)


def resolve_binding(
    protocol: CompiledProtocol,
    binding: CandidateBinding,
    *,
    authenticator: ProducerAuthenticator | None = None,
) -> BindingResult:
    """Resolve which required slots are present and which are authenticated.

    The income component slot is required only when the income block is claimed,
    exactly as the document conditions it. Every other slot is unconditional.
    Without an injected authenticator nothing is authenticated, so the mapping
    stays ``unapproved`` and every declared slot is reported unauthenticated —
    presence is never promoted to approval here.
    """
    declared = {*IDENTITY_SLOTS, *LINEAGE_SLOTS, *MAPPING_SLOTS, *EXPOSURE_SLOTS}
    if declared != set(protocol.required_binding_slots):
        raise BindingError(
            "The slot partition has drifted from the document: "
            f"only in code {sorted(declared - set(protocol.required_binding_slots))}, "
            f"only in document {sorted(set(protocol.required_binding_slots) - declared)}."
        )
    required = [
        name
        for name in protocol.required_binding_slots
        if binding.income_claimed
        or name != "income_eight_component_definition_id_and_sha256_when_income_claimed"
    ]
    unknown = tuple(sorted(set(binding.slots) - set(protocol.required_binding_slots)))
    missing = tuple(name for name in required if name not in binding.slots)
    present = [name for name in required if name in binding.slots]
    if authenticator is None:
        unauthenticated = tuple(present)
    else:
        unauthenticated = tuple(
            name
            for name in present
            if not authenticator.authenticates(name, binding.slots[name])
        )
    problems = [_SLOT_REASONS[name] for name in (*missing, *unauthenticated)]
    if unknown:
        problems.append(Reason.MAPPING_UNAPPROVED)
    if authenticator is None:
        problems.append(Reason.MAPPING_UNAPPROVED)
    status = MappingStatus.UNAPPROVED if problems else MappingStatus.APPROVED
    return BindingResult(
        status=status,
        missing_slots=missing,
        unauthenticated_slots=unauthenticated,
        unknown_slots=unknown,
        reasons=dedupe(problems),
        income_claimed=binding.income_claimed,
        evidence=MappingProxyType(
            {
                "authenticator_injected": authenticator is not None,
                "presence_is_caller_evidence_not_authentication": True,
                "income_claimed": binding.income_claimed,
                "required_slot_count": len(required),
            }
        ),
    )
