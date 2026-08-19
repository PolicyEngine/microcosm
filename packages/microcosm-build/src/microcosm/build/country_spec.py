"""The declarative country spec: one loader for a spec-only country package.

A country package (``microcosm/build/<country>/``) is pure data — JSON specs
validated by shared schema classes, executed only through shared runtimes.
This module is the schema for the package as a whole: it loads
``country_package.json`` plus every typed resource, refuses undeclared or
malformed resources, content-hashes each resource, and compiles the source
manifest into a :class:`~microcosm.build.plan.StagePlan` with the same
no-fallback posture as the US plan (an implementation per declared stage or
nothing).

Belgium is the first full consumer (microcosm#261, epic microcosm#259): the
``be/`` package declares its BE-SILC source stages, clone-and-assign commune
geography spine (vintage-aware NIS codes), Ledger target references (never
values), gate selection with criticality tiers, and the release contract —
zero Python in the country folder.

Typed resources, by canonical filename:

- ``source_stages.json`` — :class:`~microcosm.build.source_manifest.SourceManifest`
- ``support_spine.json`` — :class:`~microcosm.build.source_manifest.SupportSpineManifest`
- ``geography_spine.json`` — :class:`GeographySpineManifest` (this module)
- ``target_references.json`` — Ledger references, validated as
  :class:`~microcosm.build.ledger_targets.LedgerTargetReference` rows
- ``gates.json`` — :class:`GatesManifest` (this module)
- ``release_contract.json`` — :class:`ReleaseContractManifest` (this module)

Any other declared resource must still be a JSON mapping (the spec-only
tests forbid everything else), but its interpretation belongs to the runtime
that reads it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import resources as importlib_resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.plan import DonorSpec, Stage, StagePlan
from microcosm.build.source_manifest import (
    SourceManifest,
    SupportSpineManifest,
)
from microcosm.build.trace import compute_composition_fingerprint, sha256_file
from microcosm.frame import Frame

__all__ = [
    "ALLOWED_GATE_FUNCTIONS",
    "ALLOWED_GATE_CRITICALITIES",
    "ALLOWED_GATE_PHASES",
    "ALLOWED_GEOGRAPHY_SPINE_METHODS",
    "CountrySpec",
    "GateSelectionSpec",
    "GatesManifest",
    "GeographySpineManifest",
    "GeographySpineSpec",
    "ReleaseContractManifest",
    "country_stage_plan",
    "load_country_spec",
]

#: Gate functions a gates spec may select — the release-gate vocabulary of
#: :mod:`microcosm.build.gates`. Incumbent-comparison gates (parity,
#: export_surface, target_surface) are listed too: a first-party country
#: selects them against its frozen reference; a greenfield country like
#: Belgium simply does not select them (there is no incumbent — external
#: oracles replace self-parity, see microcosm#264).
ALLOWED_GATE_FUNCTIONS = frozenset(
    {
        "aggregate_admin",
        "degenerate_release_surface",
        "enum_domain",
        "export_surface",
        "exported_nonzero",
        "formula_owned_export",
        "input_mass_parity",
        "macro_realism",
        "nonconstant_columns",
        "nonnegative_columns",
        "parity",
        "per_family_fit",
        "release_input_coverage",
        "source_coverage",
        "spine_agreement",
        "support",
        "tail_concentration",
        "take_up_signal",
        "target_fit",
        "target_profile_coverage",
        "target_surface",
        "weight_ess",
        "weight_ratio",
        "weights_audit",
        "zero_weight_strata",
    }
)

ALLOWED_GATE_CRITICALITIES = frozenset({"release_blocking", "diagnostic"})

#: The complete key vocabulary of one ``gates.json`` gate entry. Entries are
#: validated closed-world: an unknown key is refused rather than ignored,
#: because a silently dropped key (a typo'd ``parameters``, a speculative
#: extension) would run the gate on defaults while the declared intent
#: vanished from ``policy_sha256`` — an unattested threshold.
_GATE_ENTRY_KEYS = frozenset(
    {
        "id",
        "gate",
        "phase",
        "criticality",
        "parameters",
        "not_applicable",
        "evidence_absent_blocks",
        "notes",
    }
)

#: Build phases a gate selection may bind to — the shared vocabulary that
#: keeps gate reports comparable across countries. The *order* phases run in
#: is country data (the ``phases`` header of ``gates.json``), never a global
#: constant: the UK national build gates at ``preflight`` and ``terminal``,
#: while the US pool builder gates at its checkpoint boundaries
#: (``assembled``/``transferred``/``simulated``). Same shape as
#: ``SourceOperatorContract.phases`` in the pool builder, applied to gates
#: instead of operators (microcosm#611).
ALLOWED_GATE_PHASES = frozenset(
    {"preflight", "assembled", "transferred", "simulated", "terminal"}
)

#: Geography-spine construction methods shared runtimes implement. One today:
#: uniform clone-and-assign over the declared geography level (the UK OA
#: pattern, generalized).
ALLOWED_GEOGRAPHY_SPINE_METHODS = frozenset({"clone_assign_uniform"})

#: Geography vintage-mismatch policies. Only ``"error"`` is allowed: a
#: commune-grain target bound to one NIS vintage joined against a spine of
#: another must fail compilation, never silently partial-join (the
#: microcosm#205 lesson).
ALLOWED_VINTAGE_POLICIES = frozenset({"error"})

#: Lookahead rather than \b so embedded tokens ("populace_be_v2_staging")
#: are caught, not just trailing ones; "sha-v2x" and "nuts1" stay benign.
_ORDINAL_VERSION_PATTERN = re.compile(r"[-_]v\d+(?=[-_.]|$)")

#: Observed-value keys are refused at every nesting depth: a value smuggled
#: under ledger_selector or metadata is as much a Ledger bypass as one at
#: the top level.
_FORBIDDEN_VALUE_KEYS = frozenset({"value", "values", "observed", "observed_value"})


def _carried_value_keys(node: Any) -> set[str]:
    carried: set[str] = set()
    if isinstance(node, Mapping):
        carried.update(_FORBIDDEN_VALUE_KEYS.intersection(node))
        for child in node.values():
            carried.update(_carried_value_keys(child))
    elif isinstance(node, (list, tuple)):
        for child in node:
            carried.update(_carried_value_keys(child))
    return carried


def _require_non_empty_string(value: Any, *, field_name: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field_name} must be a non-empty string.")
    return value


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected a JSON object.")
    return value


def _freeze_gate_parameter(value: Any, *, path: str) -> Any:
    """Return a recursively immutable copy of one gate parameter value."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings, got {key!r}.")
            frozen[key] = _freeze_gate_parameter(child, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_gate_parameter(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        return frozenset(
            _freeze_gate_parameter(child, path=f"{path}[]") for child in value
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"{path} must contain only mappings, sequences, sets, and JSON scalar "
        f"values; got {type(value).__name__}."
    )


def _require_header(
    raw: Mapping[str, Any], *, resource: str, country: str | None
) -> tuple[str, int, str]:
    """Validate the shared manifest header (country, version, policy)."""
    declared_country = _require_non_empty_string(
        raw.get("country"), field_name="country", context=resource
    )
    if country is not None and declared_country != country:
        raise ValueError(
            f"{resource}: declares country {declared_country!r} but lives in "
            f"the {country!r} package."
        )
    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{resource}: version must be an integer >= 1.")
    policy = _require_non_empty_string(
        raw.get("policy"), field_name="policy", context=resource
    )
    return declared_country, version, policy


# ---------------------------------------------------------------------------
# Geography spine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeographySpineSpec:
    """Clone-and-assign geography spine, declared as data.

    The spine is the standing local-area architecture: one national dataset
    carrying its finest geography as spine columns, never per-area files.
    Every assignment declares the code system *and its vintage* — municipal
    code sets change across years (Belgian NIS mergers, US congressional
    redistricting), and a vintage mismatch between spine and targets is a
    compile error, not a silent partial join.

    Attributes:
        stage: Stage name the spine construction runs as.
        method: One of :data:`ALLOWED_GEOGRAPHY_SPINE_METHODS`.
        geography_level: The finest assigned grain (e.g. ``"commune"``).
        code_system: Code registry the spine columns use (e.g. ``"be_nis"``).
        code_column: Spine column carrying the assigned code.
        vintage: Code-set vintage the assignment uses (e.g. ``"2025"``).
        vintage_policy: Mismatch policy; only ``"error"`` is allowed.
        clones_per_record: Pool multiplier for the clone pool (>= 1).
        collision_avoidance: Whether clones of one source record are kept
            out of the same assigned area.
        constrain_to_column: Source column that constrains assignment (e.g.
            the SILC NUTS1 region — clones only receive communes inside
            their observed region). Empty means unconstrained.
        assignment_source: Survey/citation for the assignment distribution
            (e.g. Statbel population by commune).
        notes: Free-text caveats.
    """

    stage: str
    method: str
    geography_level: str
    code_system: str
    code_column: str
    vintage: str
    vintage_policy: str
    clones_per_record: int
    collision_avoidance: bool
    constrain_to_column: str
    assignment_source: DonorSpec
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> GeographySpineSpec:
        context = "geography_spine"
        raw = _require_mapping(raw, context=context)
        method = _require_non_empty_string(
            raw.get("method"), field_name="method", context=context
        )
        if method not in ALLOWED_GEOGRAPHY_SPINE_METHODS:
            raise ValueError(
                f"{context}: unknown method {method!r}; allowed: "
                f"{sorted(ALLOWED_GEOGRAPHY_SPINE_METHODS)}."
            )
        vintage_policy = _require_non_empty_string(
            raw.get("vintage_policy"), field_name="vintage_policy", context=context
        )
        if vintage_policy not in ALLOWED_VINTAGE_POLICIES:
            raise ValueError(
                f"{context}: vintage_policy must be one of "
                f"{sorted(ALLOWED_VINTAGE_POLICIES)}; a silent partial join on "
                f"mismatched geography vintages is the failure this field "
                f"exists to forbid."
            )
        clones = raw.get("clones_per_record")
        if not isinstance(clones, int) or isinstance(clones, bool) or clones < 1:
            raise ValueError(f"{context}: clones_per_record must be an integer >= 1.")
        collision = raw.get("collision_avoidance")
        if not isinstance(collision, bool):
            raise ValueError(f"{context}: collision_avoidance must be a boolean.")
        source_raw = _require_mapping(
            raw.get("assignment_source"), context=f"{context}.assignment_source"
        )
        assignment_source = DonorSpec(
            survey=_require_non_empty_string(
                source_raw.get("survey"), field_name="survey", context=context
            ),
            source=_require_non_empty_string(
                source_raw.get("source"), field_name="source", context=context
            ),
            notes=str(source_raw.get("notes", "")),
        )
        return cls(
            stage=_require_non_empty_string(
                raw.get("stage"), field_name="stage", context=context
            ),
            method=method,
            geography_level=_require_non_empty_string(
                raw.get("geography_level"),
                field_name="geography_level",
                context=context,
            ),
            code_system=_require_non_empty_string(
                raw.get("code_system"), field_name="code_system", context=context
            ),
            code_column=_require_non_empty_string(
                raw.get("code_column"), field_name="code_column", context=context
            ),
            vintage=_require_non_empty_string(
                raw.get("vintage"), field_name="vintage", context=context
            ),
            vintage_policy=vintage_policy,
            clones_per_record=clones,
            collision_avoidance=collision,
            constrain_to_column=str(raw.get("constrain_to_column", "")),
            assignment_source=assignment_source,
            notes=str(raw.get("notes", "")),
        )


@dataclass(frozen=True)
class GeographySpineManifest:
    """Country geography-spine manifest (``geography_spine.json``)."""

    country: str
    version: int
    policy: str
    geography_spine: GeographySpineSpec

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, country: str | None = None
    ) -> GeographySpineManifest:
        raw = _require_mapping(raw, context="geography_spine.json")
        declared, version, policy = _require_header(
            raw, resource="geography_spine.json", country=country
        )
        return cls(
            country=declared,
            version=version,
            policy=policy,
            geography_spine=GeographySpineSpec.from_mapping(
                _require_mapping(
                    raw.get("geography_spine"),
                    context="geography_spine.json: geography_spine",
                )
            ),
        )


# ---------------------------------------------------------------------------
# Gate selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateSelectionSpec:
    """One selected release gate with its phase and criticality tier.

    Attributes:
        id: Unique id of this selection (one gate function may be selected
            more than once with different parameters, e.g. an admin gate per
            target family).
        gate: The gate function, from :data:`ALLOWED_GATE_FUNCTIONS`.
        phase: The build phase the gate evaluates in, from
            :data:`ALLOWED_GATE_PHASES` and a member of the country's
            declared phase order. Gates batch within a phase and the build
            fails closed at the phase boundary, so a gate sits where its
            evidence appears without giving up the single batched report.
        criticality: ``"release_blocking"`` or ``"diagnostic"``.
        parameters: Declarative parameters the gate runner interprets
            (tolerances, surfaces, exemption lists). Pure data. Thresholds
            live here — per country by construction, covered by the
            battery's policy hash — never in runner code.
        not_applicable: Reviewed reason this gate deliberately does not run
            for this country yet (e.g. no take-up assignment surface), or
            ``None``. A declared entry turns a silent gap into a receipt:
            it appears in every report as ``not_applicable`` and never
            evaluates. Mutually exclusive with ``parameters`` — an excused
            gate with tuned thresholds is a contradiction.
        evidence_absent_blocks: When true, an ``evidence_absent`` outcome on
            this entry blocks the build in every posture, not only under the
            release-candidate posture. For entries whose declared intent is
            that absence is never excusable (e.g. "an absent audit is not a
            passing audit") — the outcome stays honestly ``evidence_absent``
            in the report; only the enforcement changes. Meaningless on an
            excused entry, so mutually exclusive with ``not_applicable``.
        notes: Free-text rationale.
    """

    id: str
    gate: str
    phase: str
    criticality: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    not_applicable: str | None = None
    evidence_absent_blocks: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, Mapping):
            raise TypeError(
                "GateSelectionSpec parameters must be a mapping, got "
                f"{type(self.parameters).__name__}."
            )
        if not isinstance(self.evidence_absent_blocks, bool):
            raise TypeError(
                "GateSelectionSpec evidence_absent_blocks must be a bool, got "
                f"{type(self.evidence_absent_blocks).__name__}."
            )
        object.__setattr__(
            self,
            "parameters",
            _freeze_gate_parameter(
                self.parameters, path=f"gate {self.id!r}.parameters"
            ),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> GateSelectionSpec:
        raw = _require_mapping(raw, context="gates.json gate entry")
        unknown = sorted(set(raw) - _GATE_ENTRY_KEYS)
        if unknown:
            label = raw.get("id")
            entry_context = (
                f"gates.json gate entry {label!r}"
                if isinstance(label, str) and label
                else "gates.json gate entry"
            )
            raise ValueError(
                f"{entry_context} has unknown keys "
                f"{unknown}; allowed: {sorted(_GATE_ENTRY_KEYS)}. A silently "
                "dropped key would ship outside the policy hash."
            )
        gate_id = _require_non_empty_string(
            raw.get("id"), field_name="id", context="gates.json gate entry"
        )
        gate = _require_non_empty_string(
            raw.get("gate"), field_name="gate", context=f"gate {gate_id!r}"
        )
        if gate not in ALLOWED_GATE_FUNCTIONS:
            raise ValueError(
                f"gate {gate_id!r}: unknown gate function {gate!r}; allowed: "
                f"{sorted(ALLOWED_GATE_FUNCTIONS)}."
            )
        phase = _require_non_empty_string(
            raw.get("phase"), field_name="phase", context=f"gate {gate_id!r}"
        )
        if phase not in ALLOWED_GATE_PHASES:
            raise ValueError(
                f"gate {gate_id!r}: unknown phase {phase!r}; allowed: "
                f"{sorted(ALLOWED_GATE_PHASES)}."
            )
        criticality = _require_non_empty_string(
            raw.get("criticality"),
            field_name="criticality",
            context=f"gate {gate_id!r}",
        )
        if criticality not in ALLOWED_GATE_CRITICALITIES:
            raise ValueError(
                f"gate {gate_id!r}: criticality must be one of "
                f"{sorted(ALLOWED_GATE_CRITICALITIES)}, got {criticality!r}."
            )
        parameters = raw.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError(f"gate {gate_id!r}: parameters must be a JSON object.")
        not_applicable = raw.get("not_applicable")
        if not_applicable is not None:
            not_applicable = _require_non_empty_string(
                not_applicable,
                field_name="not_applicable",
                context=f"gate {gate_id!r}",
            )
            if parameters:
                raise ValueError(
                    f"gate {gate_id!r}: not_applicable and parameters are "
                    "mutually exclusive — an excused gate with tuned "
                    "thresholds is a contradiction."
                )
        evidence_absent_blocks = raw.get("evidence_absent_blocks", False)
        if not isinstance(evidence_absent_blocks, bool):
            raise ValueError(
                f"gate {gate_id!r}: evidence_absent_blocks must be a JSON "
                f"boolean, got {evidence_absent_blocks!r}."
            )
        if evidence_absent_blocks and not_applicable is not None:
            raise ValueError(
                f"gate {gate_id!r}: evidence_absent_blocks and not_applicable "
                "are mutually exclusive — an excused entry never evaluates, "
                "so demanding its absence block is a contradiction."
            )
        return cls(
            id=gate_id,
            gate=gate,
            phase=phase,
            criticality=criticality,
            parameters=dict(parameters),
            not_applicable=not_applicable,
            evidence_absent_blocks=evidence_absent_blocks,
            notes=str(raw.get("notes", "")),
        )


@dataclass(frozen=True)
class GatesManifest:
    """Country gate selection (``gates.json``).

    Attributes:
        phases: The country's phase order, first to last. Declared per
            country because build boundaries differ (the UK national build
            has two, the US pool builder three); the ``unreached`` outcome
            in a gate battery report is only computable because the report
            knows this declared order.

    Raises:
        ValueError: On duplicate selection ids, no release-blocking gate —
            a country whose every gate is diagnostic has no release
            contract — or a gate bound to a phase outside the declared
            order.
    """

    country: str
    version: int
    policy: str
    phases: tuple[str, ...]
    gates: tuple[GateSelectionSpec, ...]

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, country: str | None = None
    ) -> GatesManifest:
        raw = _require_mapping(raw, context="gates.json")
        declared, version, policy = _require_header(
            raw, resource="gates.json", country=country
        )
        phases_field = raw.get("phases")
        if not isinstance(phases_field, list) or not phases_field:
            raise ValueError(
                "gates.json: phases must be a non-empty list declaring the "
                "country's phase order."
            )
        phases = tuple(
            _require_non_empty_string(
                phase, field_name="phases entry", context="gates.json"
            )
            for phase in phases_field
        )
        unknown_phases = sorted(set(phases) - ALLOWED_GATE_PHASES)
        if unknown_phases:
            raise ValueError(
                f"gates.json: unknown phase(s) {unknown_phases}; allowed: "
                f"{sorted(ALLOWED_GATE_PHASES)}."
            )
        if len(set(phases)) != len(phases):
            duplicated = sorted({name for name in phases if phases.count(name) > 1})
            raise ValueError(f"gates.json: duplicate phase(s) {duplicated}.")
        entries = raw.get("gates")
        if not isinstance(entries, list) or not entries:
            raise ValueError("gates.json: gates must be a non-empty list.")
        gates = tuple(GateSelectionSpec.from_mapping(entry) for entry in entries)
        ids = [gate.id for gate in gates]
        if len(set(ids)) != len(ids):
            duplicated = sorted({name for name in ids if ids.count(name) > 1})
            raise ValueError(f"gates.json: duplicate gate id(s) {duplicated}.")
        undeclared = sorted({gate.phase for gate in gates if gate.phase not in phases})
        if undeclared:
            raise ValueError(
                f"gates.json: gate phase(s) {undeclared} are not in the "
                f"declared phase order {list(phases)}."
            )
        if not any(gate.criticality == "release_blocking" for gate in gates):
            raise ValueError(
                "gates.json: at least one gate must be release_blocking; a "
                "country with only diagnostic gates has no release contract."
            )
        return cls(
            country=declared,
            version=version,
            policy=policy,
            phases=phases,
            gates=gates,
        )


# ---------------------------------------------------------------------------
# Release contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseContractManifest:
    """Country release contract (``release_contract.json``).

    Declares where releases publish and what a release must contain — the
    data the publish path validates against. Naming follows the
    data_build_id convention (builder + build sha + date); ordinal version
    tokens (``-v1``, ``_v2``) anywhere in templates or filenames are
    refused.

    Attributes:
        country: Country code.
        version: Manifest schema version.
        policy: The package's standing policy line.
        builder: data_build_id builder prefix (e.g. ``"populace-be"``).
        artifact_repo: Hugging Face dataset repo holding release artifacts.
        artifact_repo_private: Whether that repo is private. A restricted
            licence requires it.
        staging_repo: Staging telemetry repo (dashboard monitor).
        dataset_filename_template: Engine-native dataset filename with a
            ``{year}`` placeholder, ending in ``.h5``.
        required_release_files: Artifacts every release must ship (the
            contract-validation list).
        public_artifacts: Release files safe for public diagnostics.
        private_artifacts: Files that never leave the private repo.
        licence_name: The microdata licence the boundary answers to.
        licence_restricted: Whether the microdata licence is restricted.
        notes: Free-text caveats.
    """

    country: str
    version: int
    policy: str
    builder: str
    artifact_repo: str
    artifact_repo_private: bool
    staging_repo: str
    dataset_filename_template: str
    required_release_files: tuple[str, ...]
    public_artifacts: tuple[str, ...]
    private_artifacts: tuple[str, ...]
    licence_name: str
    licence_restricted: bool
    notes: str = ""

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, country: str | None = None
    ) -> ReleaseContractManifest:
        context = "release_contract.json"
        raw = _require_mapping(raw, context=context)
        declared, version, policy = _require_header(
            raw, resource=context, country=country
        )
        builder = _require_non_empty_string(
            raw.get("builder"), field_name="builder", context=context
        )
        hf = _require_mapping(raw.get("hf"), context=f"{context}: hf")
        artifact_repo = _require_non_empty_string(
            hf.get("artifact_repo"), field_name="hf.artifact_repo", context=context
        )
        staging_repo = _require_non_empty_string(
            hf.get("staging_repo"), field_name="hf.staging_repo", context=context
        )
        private = hf.get("private")
        if not isinstance(private, bool):
            raise ValueError(f"{context}: hf.private must be a boolean.")
        licence = _require_mapping(raw.get("licence"), context=f"{context}: licence")
        licence_name = _require_non_empty_string(
            licence.get("name"), field_name="licence.name", context=context
        )
        restricted = licence.get("restricted")
        if not isinstance(restricted, bool):
            raise ValueError(f"{context}: licence.restricted must be a boolean.")
        if restricted and not private:
            raise ValueError(
                f"{context}: a restricted microdata licence "
                f"({licence_name!r}) requires a private artifact repo; "
                "hf.private is false."
            )
        template = _require_non_empty_string(
            raw.get("dataset_filename_template"),
            field_name="dataset_filename_template",
            context=context,
        )
        if "{year}" not in template or not template.endswith(".h5"):
            raise ValueError(
                f"{context}: dataset_filename_template must contain "
                "'{year}' and end with '.h5'."
            )
        required = raw.get("required_release_files")
        if not isinstance(required, list) or not required:
            raise ValueError(
                f"{context}: required_release_files must be a non-empty list."
            )
        boundary = _require_mapping(raw.get("boundary"), context=f"{context}: boundary")
        public = tuple(str(name) for name in boundary.get("public", ()))
        private_artifacts = tuple(str(name) for name in boundary.get("private", ()))
        if restricted and not private_artifacts:
            raise ValueError(
                f"{context}: a restricted licence must name its private "
                "artifacts (the files that never leave the private repo)."
            )
        for value in (builder, artifact_repo, staging_repo, template, *required):
            if _ORDINAL_VERSION_PATTERN.search(str(value)):
                raise ValueError(
                    f"{context}: ordinal version token in {value!r}; releases "
                    "are identified by data_build_id (builder + sha + date) "
                    "and HF revisions, never ordinals."
                )
        return cls(
            country=declared,
            version=version,
            policy=policy,
            builder=builder,
            artifact_repo=artifact_repo,
            artifact_repo_private=private,
            staging_repo=staging_repo,
            dataset_filename_template=template,
            required_release_files=tuple(str(name) for name in required),
            public_artifacts=public,
            private_artifacts=private_artifacts,
            licence_name=licence_name,
            licence_restricted=restricted,
            notes=str(raw.get("notes", "")),
        )


# ---------------------------------------------------------------------------
# Target references resource
# ---------------------------------------------------------------------------


def _validate_target_references(
    raw: Mapping[str, Any], *, country: str
) -> tuple[LedgerTargetReference, ...]:
    """Validate a ``target_references.json`` resource.

    Targets arrive by reference only: each row must parse as a
    :class:`LedgerTargetReference` (fact key / source record id / selector),
    and no row may carry an observed value — values live in Ledger.
    """
    context = "target_references.json"
    declared = _require_non_empty_string(
        raw.get("country"), field_name="country", context=context
    )
    if declared != country:
        raise ValueError(
            f"{context}: declares country {declared!r} but lives in the "
            f"{country!r} package."
        )
    rows = raw.get("target_references")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{context}: target_references must be a non-empty list.")
    references = []
    for row in rows:
        row = _require_mapping(row, context=f"{context} reference")
        carried = sorted(_carried_value_keys(row))
        if carried:
            raise ValueError(
                f"{context}: reference {row.get('name')!r} carries observed "
                f"value key(s) {carried}; targets are references — values "
                "live in Ledger, never in microcosm."
            )
        try:
            references.append(LedgerTargetReference(**row))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{context}: reference {row.get('name')!r} is invalid: {error}"
            ) from error
    return tuple(references)


# ---------------------------------------------------------------------------
# The country spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CountrySpec:
    """A loaded, validated, content-hashed country package.

    Attributes:
        country: Country code (directory name and every resource's
            ``country`` field).
        policy: The package's standing policy line.
        resources: Declared resource filenames, in manifest order.
        sources: The source-stage manifest, when declared.
        support_spine: The support-spine manifest, when declared.
        geography_spine: The geography-spine manifest, when declared.
        target_references: Ledger target references, when declared.
        gates: The gate selection, when declared.
        release_contract: The release contract, when declared.
        resource_hashes: Resource filename -> sha256 of its bytes (includes
            ``country_package.json`` itself).
    """

    country: str
    policy: str
    resources: tuple[str, ...]
    sources: SourceManifest | None
    support_spine: SupportSpineManifest | None
    geography_spine: GeographySpineManifest | None
    target_references: tuple[LedgerTargetReference, ...]
    gates: GatesManifest | None
    release_contract: ReleaseContractManifest | None
    resource_hashes: Mapping[str, str]

    @property
    def fingerprint(self) -> str:
        """Composition fingerprint over every resource hash.

        Two checkouts with byte-identical specs share a fingerprint; any
        edited resource changes it. Release manifests record it so a build
        names exactly the spec content that produced it.
        """
        return compute_composition_fingerprint(self.resource_hashes.values())


def load_country_spec(country: str | Path) -> CountrySpec:
    """Load and validate the ``country`` package.

    Args:
        country: A country code (resolved inside ``microcosm.build``) or a
            path to a country package directory.

    Returns:
        The validated :class:`CountrySpec`.

    Raises:
        FileNotFoundError: If the package or a declared resource is missing.
        ValueError: On any malformed resource (the message names the file
            and field), an undeclared file (everything in the folder must be
            declared), or a country mismatch.
    """
    if isinstance(country, Path):
        root = country
    else:
        root = Path(str(importlib_resources.files("microcosm.build").joinpath(country)))
    package_path = root / "country_package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"No country package at {package_path}.")
    package = _require_mapping(
        json.loads(package_path.read_text(encoding="utf-8")),
        context="country_package.json",
    )
    declared_country = _require_non_empty_string(
        package.get("country"), field_name="country", context="country_package.json"
    )
    if declared_country != root.name:
        raise ValueError(
            f"country_package.json declares country {declared_country!r} but "
            f"lives in directory {root.name!r}."
        )
    policy = _require_non_empty_string(
        package.get("policy"), field_name="policy", context="country_package.json"
    )
    resources_field = package.get("resources", [])
    if not isinstance(resources_field, list):
        raise ValueError("country_package.json: resources must be a list.")
    resources = tuple(str(name) for name in resources_field)

    on_disk = {path.name for path in root.iterdir() if path.is_file()}
    undeclared = sorted(on_disk - set(resources) - {"country_package.json"})
    if undeclared:
        raise ValueError(
            f"{root.name}: file(s) {undeclared} are not declared in "
            "country_package.json resources; a country package declares "
            "everything it ships."
        )

    hashes: dict[str, str] = {"country_package.json": sha256_file(package_path)}
    payloads: dict[str, Mapping[str, Any]] = {}
    for name in resources:
        resource_path = root / name
        if not resource_path.exists():
            raise FileNotFoundError(
                f"{root.name}: declared resource {name!r} is missing."
            )
        hashes[name] = sha256_file(resource_path)
        payloads[name] = _require_mapping(
            json.loads(resource_path.read_text(encoding="utf-8")), context=name
        )

    sources = (
        SourceManifest.from_mapping(payloads["source_stages.json"])
        if "source_stages.json" in payloads
        else None
    )
    if sources is not None and sources.country != declared_country:
        raise ValueError(
            f"source_stages.json: declares country {sources.country!r} but "
            f"lives in the {declared_country!r} package."
        )
    support_spine = (
        SupportSpineManifest.from_mapping(payloads["support_spine.json"])
        if "support_spine.json" in payloads
        else None
    )
    if support_spine is not None and support_spine.country != declared_country:
        raise ValueError(
            f"support_spine.json: declares country {support_spine.country!r} "
            f"but lives in the {declared_country!r} package."
        )
    geography_spine = (
        GeographySpineManifest.from_mapping(
            payloads["geography_spine.json"], country=declared_country
        )
        if "geography_spine.json" in payloads
        else None
    )
    target_references = (
        _validate_target_references(
            payloads["target_references.json"], country=declared_country
        )
        if "target_references.json" in payloads
        else ()
    )
    gates = (
        GatesManifest.from_mapping(payloads["gates.json"], country=declared_country)
        if "gates.json" in payloads
        else None
    )
    release_contract = (
        ReleaseContractManifest.from_mapping(
            payloads["release_contract.json"], country=declared_country
        )
        if "release_contract.json" in payloads
        else None
    )

    return CountrySpec(
        country=declared_country,
        policy=policy,
        resources=resources,
        sources=sources,
        support_spine=support_spine,
        geography_spine=geography_spine,
        target_references=target_references,
        gates=gates,
        release_contract=release_contract,
        resource_hashes=hashes,
    )


def country_stage_plan(
    spec: CountrySpec,
    implementations: Mapping[str, Callable[[Frame], Frame]],
    stage_names: tuple[str, ...] | None = None,
) -> StagePlan:
    """Assemble the country's :class:`StagePlan` from its source manifest.

    The US plan's no-fallback posture, generalized: every declared stage
    (source stages in manifest order, then the geography-spine stage when
    declared) needs exactly one implementation by default; callers may
    select a validated non-empty subset for country pipelines that share one
    manifest while running separate entry points.

    Args:
        spec: A loaded country spec with a source manifest.
        implementations: One ``transform(frame) -> Frame`` per declared
            selected stage.
        stage_names: Optional declared-stage names to run. Selection is
            validated against the manifest, and execution order remains the
            manifest order regardless of this tuple's order. Implementations
            for declared-but-unselected stages are deliberately tolerated —
            pipelines sharing one implementations map may each select their
            own subset — so an extra entry for an unselected stage is a
            silent no-op, not an error; only implementations for stages the
            manifest never declares are refused.

    Returns:
        The validated plan, each stage carrying its manifest citation as
        the donor record.

    Raises:
        ValueError: If the spec declares no source stages, a requested
            selection is empty or unknown, an implementation is missing for a
            selected stage, or an implementation is supplied for an
            undeclared stage.
    """
    if spec.sources is None:
        raise ValueError(
            f"Country {spec.country!r} declares no source_stages.json; there "
            "is no stage plan to assemble."
        )
    declared: list[tuple[str, DonorSpec | None, tuple[str, ...]]] = [
        (
            stage.stage,
            DonorSpec(survey=stage.survey, source=stage.source, notes=stage.notes),
            stage.outputs,
        )
        for stage in spec.sources.stages
    ]
    if spec.geography_spine is not None:
        spine = spec.geography_spine.geography_spine
        declared.append(
            (
                spine.stage,
                DonorSpec(
                    survey=spine.assignment_source.survey,
                    source=spine.assignment_source.source,
                    notes=spine.assignment_source.notes,
                ),
                (spine.code_column,),
            )
        )
    declared_names = [name for name, _, _ in declared]
    selected_names: tuple[str, ...]
    if stage_names is None:
        selected_names = tuple(declared_names)
    else:
        if not stage_names:
            raise ValueError("country_stage_plan stage_names must not be empty.")
        duplicates = sorted(
            {name for name in stage_names if stage_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(
                f"country_stage_plan stage_names contains duplicate stage(s) "
                f"{duplicates}."
            )
        unknown_selection = sorted(set(stage_names) - set(declared_names))
        if unknown_selection:
            raise ValueError(
                f"Unknown stage selection {unknown_selection}; declared stages "
                f"are {declared_names}."
            )
        requested = set(stage_names)
        selected_names = tuple(name for name in declared_names if name in requested)
    missing = [name for name in selected_names if name not in implementations]
    if missing:
        raise ValueError(
            f"country_stage_plan needs an implementation for every selected "
            f"stage; missing {missing}. There are no stubs or fallbacks by "
            "design."
        )
    unknown = sorted(set(implementations) - set(declared_names))
    if unknown:
        raise ValueError(
            f"Unknown stage implementation(s) {unknown}; declared stages are "
            f"{declared_names}."
        )
    selected = [
        (name, donor, outputs)
        for name, donor, outputs in declared
        if name in set(selected_names)
    ]
    return StagePlan(
        Stage(
            name=name,
            transform=implementations[name],
            produces=outputs,
            donor=donor,
        )
        for name, donor, outputs in selected
    )
