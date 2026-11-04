"""What UK target compilation records about its Chronicle feed, and a roster check.

``full_targets`` writes :func:`_ledger_provenance` into every compiled surface
and runs :func:`_validate_band_edge_registry` on the compiled registers. Both
lived in ``calibration_run``, the national calibration runner. The UK target
kernels hash this small module through ``UK_TARGET_COMPILE_MODULE_NAMES``
instead of all of ``calibration_run`` (microcosm#1492), so an edit to either
function re-keys the compiled surface and an edit to the runner does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from microcosm.calibrate import TargetRegistry


def _validate_band_edge_registry(
    *,
    register_registry: TargetRegistry,
    band_edge_registry: TargetRegistry,
    exclusion_receipt: Mapping[str, Mapping[str, str]],
) -> None:
    """Require the band-edge register to reconstitute the compiled roster.

    The reconciliation always runs — an empty receipt is a claim that nothing
    was pruned, so the two rosters must then be name-identical; it is never
    permission to skip the check. Receipt keys are spec names by the
    applier's construction: ``apply_uk_calibration_measure_exclusions``
    builds the receipt from the matched ``spec.name`` set and raises on any
    exclusion matching zero registry specs, so ``pruned + receipt keys ==
    compiled`` is an exact identity, not a heuristic.
    """

    register_names = _registry_spec_names(register_registry)
    edge_names = _registry_spec_names(band_edge_registry)
    excluded_names = {str(name) for name in exclusion_receipt}
    if not register_names <= edge_names:
        missing = sorted(register_names - edge_names)
        raise ValueError(
            "band-edge register must include every materialized registry spec; "
            f"missing={missing}."
        )
    extra = edge_names - register_names
    if extra != excluded_names:
        raise ValueError(
            "band-edge register extra spec names must match the measure "
            f"exclusion receipt; extra={sorted(extra)}, "
            f"receipt={sorted(excluded_names)}."
        )


def _registry_spec_names(registry: TargetRegistry) -> set[str]:
    return {str(spec.name) for spec in registry.specs}


#: The manifest fields the UK run seals into ``run_config``. Narrower than
#: the artifact's whole manifest on purpose: the identity digest should name
#: *which* published artifact was compiled, not re-hash its manifest.
_LEDGER_MANIFEST_IDENTITY_FIELDS = (
    "artifact_id",
    "profile",
    "schema_version",
    "generated_at",
)

#: Epoch witnesses :meth:`LedgerConsumerArtifact.provenance` supplies, split
#: by shape. Named here so the delegation below cannot quietly drop one: a
#: run that compiled a chronicle-era or mixed-epoch feed has to say so in its
#: own evidence, not only in the loader's return value
#: (PolicyEngine/chronicle#143).
_LEDGER_EPOCH_WITNESS_SCALARS = ("schema_epoch",)
_LEDGER_EPOCH_WITNESS_LISTS = (
    "fact_key_epochs",
    "undeclared_fact_key_domains",
    "fact_schema_versions",
)


def _ledger_provenance(artifact: Any) -> dict[str, object]:
    """The verified identity of the Chronicle consumer feed this run compiled.

    Delegates to :meth:`microcosm.build.ledger_artifact.LedgerConsumerArtifact.provenance`
    rather than rebuilding the block field by field. Rebuilding it is how the
    epoch witnesses went missing here in the first place: the loader learned
    which era resolved the targets and the UK run kept reporting only the
    hashes. Anything the shared block gains, this block gains.

    The whole block is sealed into ``run_config`` and so into the run's
    identity digest: the epoch witnesses enter it beside the hashes, which is
    why this delegation is a change to the digest of an otherwise identical
    configuration, once. Only the manifest sub-block is UK-shaped, and it
    stays narrow so the digest names the artifact rather than re-hashing its
    whole manifest.

    A bare ``consumer_facts.jsonl`` feed carries no manifest, so its
    Chronicle-side provenance is recorded as absent rather than invented. So
    is a stand-in that predates the shared block: the fields it cannot supply
    are recorded ``None``/empty rather than fabricated.
    """

    shared = getattr(artifact, "provenance", None)
    block: Mapping[str, Any] = shared() if callable(shared) else {}
    provenance: dict[str, object] = {
        field: block.get(field, getattr(artifact, field, None))
        for field in ("facts_sha256", "fact_row_count", "manifest_sha256")
    }
    for field in _LEDGER_EPOCH_WITNESS_SCALARS:
        provenance[field] = block.get(field)
    for field in _LEDGER_EPOCH_WITNESS_LISTS:
        provenance[field] = list(block.get(field) or ())
    manifest = getattr(artifact, "manifest", None)
    if isinstance(manifest, Mapping):
        provenance["manifest"] = {
            key: manifest.get(key)
            for key in _LEDGER_MANIFEST_IDENTITY_FIELDS
            if manifest.get(key) is not None
        }
    return provenance
