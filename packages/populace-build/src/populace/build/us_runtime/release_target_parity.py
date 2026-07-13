"""US release target-parity: the calibration-target surface as a HARD gate.

The target-side analog of the input-column coverage contract
(:mod:`populace.build.us_runtime.release_input_coverage`). Where that module
guards *inputs the reference eCPS exports*, this one guards *administrative
targets the retired us-data/eCPS pipeline calibrated to*.

The launch failure this closes: an administrative target family the retired
pipeline calibrated to is silently absent from the populace target registry.
The motivating case is SSA SSI recipients — the retired pipeline calibrated
``nation/ssa/ssi_recipients`` (SSA Annual Statistical Supplement, 7,404,820
recipients in 2024), the pinned consumer feed carries the
``ssa_supplement.ssi_recipients`` facts, yet the registry compiled only the six
``oasdi_ssi_payments`` aggregates and dropped the recipient family with no
record of the omission. Column *mass* parity did not catch it because a target
family that never compiles contributes zero rows, not wrong ones.

This module is the coverage contract at *family* granularity — a family is a
distinct administrative table/concept the ledger feed carries, identified by the
namespace and first concept token of a fact's ``source_record_id`` (e.g.
``ssa_supplement.ssi_recipients``, ``irs_soi.historic_table_2``,
``cbo.revenue_projection``). This is the granularity at which silent omission
happens: it is finer than the source family (``ssa`` alone would hide the
recipient gap behind the compiled OASDI payments) and coarser than the
individual row.

The versioned in-repo manifest declares every family with a status:

- ``compiled`` — the compiled registry must carry at least one target for it;
- ``reviewed_exclusion`` — a documented, classified gap allowed to be absent
  (survey-derived, macro control total, non-linear, off-by-default,
  source-absent, …), each naming its evidence.

:func:`us_release_target_parity_gate` enforces the contract on the compiled
target registry and, wired into the release tool, hard-fails the build before
materialization exactly like the target-profile coverage gate.

:func:`assert_target_parity_manifest_current` proves the manifest against the
checked-in, sha-pinned feed-family inventory (always) and the live compiled
registry (when supplied), so the register cannot silently rot (#286/#337): it
must declare exactly the feed's family surface plus any documented source-absent
us-data families, every ``compiled`` family must be in the registry, and — the
red line this contract exists for — the core SSA SSI recipient family must stay
``compiled`` and can never be quietly downgraded to a reviewed exclusion.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from populace.build.gates import GateResult

__all__ = [
    "US_TARGET_PARITY_MANIFEST_RESOURCE",
    "US_TARGET_PARITY_FEED_FAMILIES_RESOURCE",
    "COMPILED_STATUS",
    "REVIEWED_EXCLUSION_STATUS",
    "SOURCE_ABSENT_CLASSIFICATION",
    "RED_LINE_COMPILED_FAMILIES",
    "TargetFamily",
    "TargetFence",
    "TargetParityManifest",
    "assert_target_parity_manifest_current",
    "load_target_parity_feed_families",
    "load_target_parity_manifest",
    "registry_target_family_ids",
    "us_release_target_parity_compiled_families",
    "us_release_target_parity_gate",
    "us_release_target_parity_reviewed_exclusions",
    "us_target_family_id",
]

US_TARGET_PARITY_MANIFEST_RESOURCE = "target_parity_manifest.json"
US_TARGET_PARITY_FEED_FAMILIES_RESOURCE = "target_parity_feed_families.json"

_US_PACKAGE = "populace.build.us"

COMPILED_STATUS = "compiled"
REVIEWED_EXCLUSION_STATUS = "reviewed_exclusion"
_VALID_STATUSES = frozenset({COMPILED_STATUS, REVIEWED_EXCLUSION_STATUS})

#: A reviewed exclusion whose family the pinned feed does not carry at all — a
#: us-data admin target with no ledger fact to compile. These are the only
#: declared families exempt from the feed-surface reconciliation.
SOURCE_ABSENT_CLASSIFICATION = "source_absent"

#: Families that must ship ``compiled`` with no reviewed exclusion — the whole
#: point of the register is that the administrative targets the retired pipeline
#: calibrated can never again be silently dropped. The anti-rot check refuses to
#: let any of them be downgraded, mirroring the #368 SSI countable-resource asset
#: red line on the input-coverage side.
#:
#: - ``ssa_supplement.ssi_recipients``: the SSA SSI recipient count
#:   (``nation/ssa/ssi_recipients``).
#: - ``bea_nipa.total_wages_salaries``: the BEA NIPA all-population wage total
#:   (``nation/bea/nipa_wages_and_salaries``, PR #994) — economy-wide wages
#:   including nonfilers, the ~$12.4T universe that tax-return / CPS-reported
#:   wages undercount by two-thirds. Silent loss of it is the exact failure the
#:   Chesterton's-fence audit surfaced.
RED_LINE_COMPILED_FAMILIES = (
    "ssa_supplement.ssi_recipients",
    "bea_nipa.total_wages_salaries",
)


def _is_target_period_token(value: str) -> bool:
    """Whether a ``source_record_id`` token is a period tag.

    The canonical period-token test used across the ledger target machinery
    (``irs_soi.ty2023``, ``ssa_supplement.cy2024``, ``usda_snap.fy2024``, bare
    four-digit years, and ``YYYY_MM`` / ``monthYYYY_MM`` month tags). Kept in
    lockstep with
    :func:`populace.build.ledger_targets._is_period_token` so the family id is
    derived identically to the registry's own record-set normalization.
    """
    normalized = value.lower().replace("-", "_")
    if normalized.startswith("month"):
        normalized = normalized[len("month") :]
    parts = normalized.split("_", maxsplit=1)
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return len(parts[0]) == 4 and len(parts[1]) in {1, 2}
    if normalized[:2] in {"ty", "cy", "fy"}:
        normalized = normalized[2:]
    return normalized.isdigit() and len(normalized) == 4


def us_target_family_id(source_record_id: str) -> str:
    """The administrative target-family id of a Ledger ``source_record_id``.

    ``namespace.concept`` where ``namespace`` is the first token and ``concept``
    is the first following non-period token, e.g.::

        ssa_supplement.cy2024.ssi_recipients.by_area_category.al_total... ->
            ssa_supplement.ssi_recipients
        irs_soi.ty2022.historic_table_2.us.all.ctc_amount ->
            irs_soi.historic_table_2
        cbo.fy2023.revenues.individual_income_taxes.actual_amount ->
            cbo.revenues

    A registry ``TargetSpec.name`` is exactly its ``source_record_id``, so this
    maps compiled specs and raw feed facts onto the same family surface.
    """
    tokens = [token for token in str(source_record_id).split(".") if token]
    if not tokens:
        return ""
    namespace = tokens[0]
    for token in tokens[1:]:
        if not _is_target_period_token(token):
            return f"{namespace}.{token}"
    return namespace


@dataclass(frozen=True)
class TargetFence:
    """The Chesterton's-fence record behind one reviewed exclusion.

    A category label (``macro_control_total``, ``not_modeled``, ``superseded``)
    is not a sufficient reason to drop a target the retired pipeline calibrated
    to. Before an exclusion may stand it must recover *why the fence was built*:
    the target's origin, the failure mode it guarded, and the purpose-informed
    basis for not rebuilding it here. An unexplained fence gets rebuilt (wired),
    not removed.

    Attributes:
        origin: Where the fence came from — the introducing (and, where
            relevant, removing) ``us-data`` PR/commit, or the
            explicit finding ``"not a us-data calibration target"`` when the feed
            fact was never a target (a ledger reference fact).
        purpose: The failure mode the target guarded, quoting the PR/issue
            rationale verbatim where one exists, or stating ``"no discoverable
            rationale in PR #N"`` (then the exclusion must justify itself on
            mechanics), or ``"n/a"`` for a fact that was never a target.
        verdict_basis: The purpose-informed reason the exclusion stands — a
            named compiled family that subsumes it, an architecture change that
            retires the need (with a code/PR cite), or a deferral with the
            specific blocker. Never a bare category label.
    """

    origin: str
    purpose: str
    verdict_basis: str

    def __post_init__(self) -> None:
        for field_name in ("origin", "purpose", "verdict_basis"):
            if not getattr(self, field_name):
                raise ValueError(f"TargetFence.{field_name} is required.")


@dataclass(frozen=True)
class TargetFamily:
    """One declared target-parity family.

    Attributes:
        name: The ``namespace.concept`` family id.
        status: ``"compiled"`` (the registry must carry a target for it) or
            ``"reviewed_exclusion"`` (a documented, classified gap).
        classification: For a reviewed exclusion, the exclusion kind
            (``survey_derived``, ``macro_control_total``, ``non_linear``,
            ``off_by_default``, ``superseded``, ``source_absent``,
            ``deferred``, ``input_side``, ``not_modeled``, ``not_a_target``).
            A classification is now a label ON TOP of the fence narrative, not a
            reason by itself. Empty for a compiled family.
        reason: Why the gap is accepted — required for a reviewed exclusion.
        evidence: The concrete fact/mechanism the reason names (a sample
            ``source_record_id``, a code constant, a compiled sibling family).
            Required for a reviewed exclusion.
        fence: The Chesterton's-fence record (origin, purpose, verdict_basis).
            REQUIRED for every reviewed exclusion — a category label alone is no
            longer a sufficient reason to drop a us-data-era target.
        issue: Optional tracking issue owning the gap's closure.
        note: Optional free-text annotation.
    """

    name: str
    status: str
    classification: str = ""
    reason: str = ""
    evidence: str = ""
    fence: TargetFence | None = None
    issue: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TargetFamily.name is required.")
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"{self.name}: status must be one of {sorted(_VALID_STATUSES)}, "
                f"got {self.status!r}."
            )
        if self.status == REVIEWED_EXCLUSION_STATUS:
            if not self.reason:
                raise ValueError(
                    f"{self.name}: a reviewed exclusion needs a reason "
                    "(an undocumented exclusion is a silent omission)."
                )
            if not self.classification:
                raise ValueError(
                    f"{self.name}: a reviewed exclusion needs a classification."
                )
            if not self.evidence:
                raise ValueError(
                    f"{self.name}: a reviewed exclusion needs evidence naming the "
                    "concrete fact or mechanism behind the reason."
                )
            if self.fence is None:
                raise ValueError(
                    f"{self.name}: a reviewed exclusion needs a fence "
                    "{origin, purpose, verdict_basis} — a category label is not a "
                    "sufficient reason to drop a us-data-era calibration target."
                )

    @property
    def is_source_absent(self) -> bool:
        """Whether the family is excluded because the feed carries no fact."""
        return (
            self.status == REVIEWED_EXCLUSION_STATUS
            and self.classification == SOURCE_ABSENT_CLASSIFICATION
        )


@dataclass(frozen=True)
class TargetParityManifest:
    """The full parsed target-parity contract.

    Attributes:
        reference: Provenance of the feed and us-data sources the surface is
            derived from.
        families: Every declared family, in name order.
        schema_version: Manifest schema version.
    """

    reference: Mapping[str, str]
    families: tuple[TargetFamily, ...]
    schema_version: int = 1
    _by_name: dict[str, TargetFamily] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        by_name: dict[str, TargetFamily] = {}
        for family in self.families:
            if family.name in by_name:
                raise ValueError(f"Duplicate manifest family {family.name!r}.")
            by_name[family.name] = family
        object.__setattr__(self, "_by_name", by_name)

    @property
    def by_name(self) -> Mapping[str, TargetFamily]:
        return self._by_name

    @property
    def declared_families(self) -> frozenset[str]:
        """Every family the manifest declares (compiled + reviewed)."""
        return frozenset(self._by_name)

    @property
    def compiled_families(self) -> frozenset[str]:
        """Families the compiled registry must carry a target for."""
        return frozenset(
            family.name for family in self.families if family.status == COMPILED_STATUS
        )

    @property
    def reviewed_exclusions(self) -> dict[str, str]:
        """Reviewed-exclusion families as ``name -> reason``."""
        return {
            family.name: family.reason
            for family in self.families
            if family.status == REVIEWED_EXCLUSION_STATUS
        }

    @property
    def source_absent_families(self) -> frozenset[str]:
        """Reviewed exclusions the pinned feed carries no fact for."""
        return frozenset(
            family.name for family in self.families if family.is_source_absent
        )

    @property
    def feed_surface_families(self) -> frozenset[str]:
        """Declared families the pinned feed is expected to carry.

        Every declared family except the documented source-absent us-data ones;
        this must equal the checked-in feed-family inventory exactly.
        """
        return self.declared_families - self.source_absent_families


def _resource_text(resource: str) -> str:
    candidate = Path(resource)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return files(_US_PACKAGE).joinpath(resource).read_text(encoding="utf-8")


def _resource_payload(resource: str) -> Mapping[str, Any]:
    raw = json.loads(_resource_text(resource))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{resource}: expected a JSON object.")
    return raw


def load_target_parity_manifest(
    resource: str = US_TARGET_PARITY_MANIFEST_RESOURCE,
) -> TargetParityManifest:
    """Load and validate the release target-parity manifest.

    Raises:
        ValueError: If the payload shape is wrong, a family has an unknown
            status, a reviewed exclusion is missing its reason/classification/
            evidence, or the declared family set is empty (a silently-empty
            manifest would make the gate vacuous).
    """
    payload = _resource_payload(resource)

    raw_reference = payload.get("reference")
    if not isinstance(raw_reference, Mapping):
        raise ValueError(f"{resource}: 'reference' must be a JSON object.")
    reference = {str(key): str(value) for key, value in raw_reference.items()}

    raw_families = payload.get("families")
    if not isinstance(raw_families, Mapping) or not raw_families:
        raise ValueError(
            f"{resource}: 'families' must be a non-empty JSON object; a silently "
            "empty manifest would make the target-parity gate vacuous."
        )
    families: list[TargetFamily] = []
    for name, entry in sorted(raw_families.items()):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{resource}: family {name!r} must be a JSON object.")
        raw_fence = entry.get("fence")
        fence: TargetFence | None = None
        if raw_fence is not None:
            if not isinstance(raw_fence, Mapping):
                raise ValueError(
                    f"{resource}: family {name!r} 'fence' must be a JSON object."
                )
            fence = TargetFence(
                origin=str(raw_fence.get("origin", "")),
                purpose=str(raw_fence.get("purpose", "")),
                verdict_basis=str(raw_fence.get("verdict_basis", "")),
            )
        families.append(
            TargetFamily(
                name=str(name),
                status=str(entry.get("status", "")),
                classification=str(entry.get("classification", "")),
                reason=str(entry.get("reason", "")),
                evidence=str(entry.get("evidence", "")),
                fence=fence,
                issue=str(entry.get("issue", "")),
                note=str(entry.get("note", "")),
            )
        )

    schema_version = payload.get("schema_version", 1)
    if not isinstance(schema_version, int):
        raise ValueError(f"{resource}: 'schema_version' must be an integer.")

    return TargetParityManifest(
        reference=reference,
        families=tuple(families),
        schema_version=schema_version,
    )


def load_target_parity_feed_families(
    resource: str = US_TARGET_PARITY_FEED_FAMILIES_RESOURCE,
) -> Mapping[str, Any]:
    """Load the checked-in, sha-pinned feed-family inventory.

    The authoritative feed surface the manifest cannot drift from: a mapping of
    every family id the pinned consumer feed carries to its fact count, plus the
    feed filename and sha256. Derived from the feed by
    ``tools/build_us_target_parity_manifest.py`` and committed, so the manifest
    consistency check runs without the 131 MB feed in CI (the same pattern the
    input-coverage manifest uses against the checked-in eCPS parity reference).
    """
    payload = _resource_payload(resource)
    families = payload.get("families")
    if not isinstance(families, Mapping) or not families:
        raise ValueError(f"{resource}: 'families' must be a non-empty JSON object.")
    return payload


def us_release_target_parity_compiled_families() -> frozenset[str]:
    """The families the shipped manifest marks compiled."""
    return load_target_parity_manifest().compiled_families


def us_release_target_parity_reviewed_exclusions() -> dict[str, str]:
    """The reviewed-exclusion register from the shipped manifest."""
    return load_target_parity_manifest().reviewed_exclusions


def registry_target_family_ids(registry: Any) -> frozenset[str]:
    """The set of administrative target families a compiled registry carries."""
    return frozenset(
        us_target_family_id(spec.name)
        for spec in registry.specs
        if us_target_family_id(spec.name)
    )


def us_release_target_parity_gate(
    registry: Any,
    *,
    manifest: TargetParityManifest | None = None,
) -> GateResult:
    """Build the named US release target-parity gate for a compiled registry.

    Every ``compiled`` manifest family must be carried by ``registry`` as at
    least one target; a compiled family with no registry target fails the gate
    (the silent-omission failure this gate exists to catch). A reviewed
    exclusion whose family the registry now compiles is stale and fails too
    (#286/#337 cannot-rot). Run on the compiled, substituted target registry
    just before materialization, it hard-fails the release like the
    target-profile coverage gate.

    Args:
        registry: The compiled :class:`~populace.calibrate.TargetRegistry`.
        manifest: Override the shipped manifest (tests).

    Returns:
        The ``"us_release_target_parity"`` gate result.
    """
    manifest = manifest or load_target_parity_manifest()
    present = registry_target_family_ids(registry)
    compiled = manifest.compiled_families
    reviewed = set(manifest.reviewed_exclusions)

    failures: list[str] = []
    for family in sorted(compiled - present):
        failures.append(
            f"{family}: the manifest marks this administrative target family "
            "compiled, but the compiled registry carries no target for it — the "
            "silent omission this gate exists to catch. Wire it or reclassify it "
            "as a reviewed exclusion with evidence."
        )
    for family in sorted(reviewed & present):
        failures.append(
            f"{family}: the manifest marks this family a reviewed exclusion, but "
            "the registry now compiles a target for it. Promote it to compiled "
            "(the manifest cannot rot — PolicyEngine/populace#286/#337)."
        )

    return GateResult(
        name="us_release_target_parity",
        passed=not failures,
        failures=tuple(failures),
        details={
            "compiled_families": len(compiled),
            "reviewed_exclusions": len(reviewed),
            "registry_families": len(present),
        },
    )


def assert_target_parity_manifest_current(
    *,
    registry: Any | None = None,
    manifest: TargetParityManifest | None = None,
    feed_families: Mapping[str, Any] | None = None,
) -> None:
    """Fail if the target-parity manifest has drifted from its sources.

    Checked against the checked-in feed-family inventory (always) and the live
    compiled registry (when supplied):

    - The declared feed-surface families (every family except documented
      source-absent ones) must equal the checked-in feed-family inventory. A
      new family in the feed, or a manifest family the feed no longer carries,
      is drift and must be reconciled by regenerating the manifest.
    - The manifest's ``reference.feed_sha256`` must match the inventory's, so
      the two artifacts describe the same pinned feed.
    - The red-line families must stay ``compiled`` with no reviewed
      exclusion — the red line this contract exists for cannot be quietly undone.
    - When a compiled registry is supplied, every ``compiled`` family must be
      present in it, every ``reviewed_exclusion`` family must be absent from it,
      and the registry must carry no family the manifest fails to declare.

    A no-op for the registry half when no registry is supplied (the workspace
    test environment); the checked-in-facts half always runs.

    Raises:
        ValueError: Naming every drift found.
    """
    manifest = manifest or load_target_parity_manifest()
    feed_families = feed_families or load_target_parity_feed_families()
    failures: list[str] = []

    feed_family_ids = frozenset(str(name) for name in feed_families["families"])
    declared_feed_surface = manifest.feed_surface_families

    missing_from_manifest = sorted(feed_family_ids - manifest.declared_families)
    if missing_from_manifest:
        failures.append(
            "the pinned feed carries target family(ies) not declared in the "
            f"manifest {missing_from_manifest}; regenerate with "
            "tools/build_us_target_parity_manifest.py."
        )
    extra_feed_surface = sorted(declared_feed_surface - feed_family_ids)
    if extra_feed_surface:
        failures.append(
            "the manifest declares feed-surface family(ies) the pinned feed no "
            f"longer carries {extra_feed_surface}; regenerate the manifest or "
            "reclassify them source_absent."
        )

    manifest_sha = str(manifest.reference.get("feed_sha256", ""))
    inventory_sha = str(feed_families.get("feed_sha256", ""))
    if manifest_sha and inventory_sha and manifest_sha != inventory_sha:
        failures.append(
            "manifest reference.feed_sha256 "
            f"{manifest_sha!r} does not match the feed-family inventory sha "
            f"{inventory_sha!r}; regenerate both from the same pinned feed."
        )

    for family in RED_LINE_COMPILED_FAMILIES:
        entry = manifest.by_name.get(family)
        if entry is None:
            failures.append(
                f"{family}: red-line administrative target family must be declared "
                "and compiled (the target-parity red line)."
            )
        elif entry.status != COMPILED_STATUS:
            failures.append(
                f"{family}: red-line administrative target family must stay status="
                f"{COMPILED_STATUS!r}, not {entry.status!r} — it can never be "
                "quietly downgraded to a reviewed exclusion."
            )

    if registry is not None:
        present = registry_target_family_ids(registry)
        undeclared = sorted(present - manifest.declared_families)
        if undeclared:
            failures.append(
                "the compiled registry carries target family(ies) the manifest "
                f"does not declare {undeclared}; regenerate the manifest."
            )
        not_compiled = sorted(manifest.compiled_families - present)
        if not_compiled:
            failures.append(
                "the manifest marks family(ies) compiled the registry does not "
                f"carry {not_compiled}."
            )
        stale = sorted(set(manifest.reviewed_exclusions) & present)
        if stale:
            failures.append(
                "the manifest marks family(ies) a reviewed exclusion the registry "
                f"now compiles {stale}; promote them to compiled."
            )

    if failures:
        raise ValueError(
            "US release target-parity manifest has drifted:\n"
            + "\n".join(f"  - {line}" for line in failures)
        )
