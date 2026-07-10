"""US release input-column coverage: the eCPS export surface as a HARD gate.

populace #368. The launch failure this closes: an SSI asset-limit reform scored
exactly $0 on a certified default because ``ssi_countable_resources`` is 0 for
every record — the asset input columns were dropped at the base pool and are
absent from the export. This is the #340/#356/#361/#278 family: input columns
the reference eCPS exports that the sparse national release never persists, so
the engine silently defaults them and every reform binding through them scores
$0 while export-mass parity (a ~35-column weighted-mass check) still passes.
Column *mass* parity is not column *coverage*.

This module is the coverage contract. It declares, in a versioned in-repo
manifest, every input column the pinned reference eCPS populates, each with a
status:

- ``required`` — the release must persist it as a key with non-default signal;
- ``reviewed_exclusion`` — a documented, issue-tracked gap allowed to be
  absent/degenerate for now (the debt ledger that shrinks as inputs are
  restored).

:func:`us_release_input_coverage_gate` enforces it on the export frame and, wired
into the release tool, hard-fails the export before ``write_dataset`` exactly
like the export-mass parity gate. It generalizes
:func:`populace.build.us_runtime.l0_refit_export.assert_required_us_release_source_columns`
from five SNAP/ACA/immigration columns to the full eCPS input surface.

Per #368 ("it must be an actual gate"), the three SSI countable-resource asset
inputs (``bank_account_assets``, ``stock_assets``, ``bond_assets``) are
``required`` with NO reviewed exclusion even though they are currently absent, so
the gate ships RED for today's artifacts; asset restoration (Deliverable 2) turns
it green. A gate that fails the current default is the point, not a bug.

:func:`assert_release_input_coverage_manifest_current` proves the manifest against
the pinned eCPS surface and the live PolicyEngine-US graph, so the register
cannot silently rot: it must cover exactly the reference eCPS populated layers,
every declared column must be a real engine input leaf, and the SSI asset inputs
must stay hard requirements.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from populace.build.gates import GateResult, input_column_coverage_gate

__all__ = [
    "US_RELEASE_INPUT_COVERAGE_RESOURCE",
    "ReformCoverageProbe",
    "ReleaseInputColumn",
    "ReleaseInputCoverageManifest",
    "assert_release_input_coverage_manifest_current",
    "load_release_input_coverage_manifest",
    "us_release_input_coverage_gate",
    "us_release_input_coverage_required_columns",
    "us_release_input_coverage_reviewed_exclusions",
    "us_release_reform_coverage_probes",
]

US_RELEASE_INPUT_COVERAGE_RESOURCE = "release_input_coverage_manifest.json"

_US_PACKAGE = "populace.build.us"
_ECPS_PARITY_REFERENCE_RESOURCE = "ecps_parity_reference.json"

REQUIRED_STATUS = "required"
REVIEWED_EXCLUSION_STATUS = "reviewed_exclusion"
_VALID_STATUSES = frozenset({REQUIRED_STATUS, REVIEWED_EXCLUSION_STATUS})

#: The ``adds`` list of PolicyEngine-US ``ssi_countable_resources``: the asset
#: leaves SSI eligibility keys on. Per #368 these ship as hard requirements with
#: NO reviewed exclusion, so the coverage gate and the SSI reform probe both fail
#: until the asset stage is restored.
SSI_COUNTABLE_RESOURCE_ASSETS = (
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
)


@dataclass(frozen=True)
class ReleaseInputColumn:
    """One declared coverage-contract column.

    Attributes:
        name: The PolicyEngine-US input variable the release must persist.
        status: ``"required"`` (present + non-degenerate) or
            ``"reviewed_exclusion"`` (a tracked gap allowed to be absent).
        reason: Why the gap is accepted — required and non-empty for a reviewed
            exclusion, empty for a hard requirement.
        issue: The tracking issue owning the gap's closure — required for a
            reviewed exclusion.
        note: Optional free-text annotation (e.g. why an absent column is still
            a hard requirement).
    """

    name: str
    status: str
    reason: str = ""
    issue: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ReleaseInputColumn.name is required.")
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"{self.name}: status must be one of {sorted(_VALID_STATUSES)}, "
                f"got {self.status!r}."
            )
        if self.status == REVIEWED_EXCLUSION_STATUS:
            if not self.reason:
                raise ValueError(
                    f"{self.name}: a reviewed exclusion needs a reason "
                    "(an undocumented exclusion is a silent zero with extra steps)."
                )
            if not self.issue:
                raise ValueError(
                    f"{self.name}: a reviewed exclusion needs a tracking issue."
                )


@dataclass(frozen=True)
class ReformCoverageProbe:
    """A pinned reform whose $0 score on the export signals a coverage hole.

    A reform that mechanically binds through named input leaves must move its
    budget measure. If it scores ~$0 on the release, the leaves it binds through
    are absent/degenerate — the same silent-zero failure the column gate catches,
    proven end-to-end against the engine.

    Attributes:
        id: Stable probe id.
        name: Human-readable reform description.
        parameter_changes: ``Reform.from_dict`` payload (country ``us``).
        budget_measure: The variable whose weighted total change is scored.
        effect_direction: ``"reform_minus_baseline"`` or
            ``"baseline_minus_reform"``.
        period: Optional probe-specific scoring year. OBBBA probes use 2026
            while the release's source-data period remains 2024.
        expected_sign: Required sign of the direction-normalized effect.
        binding_inputs: The input leaves the reform binds through; named in the
            failure so the fix target is explicit.
        min_abs_effect: The reform fails the smoke gate when
            ``abs(effect) < min_abs_effect`` — a floor above simulation noise
            and far below the plausible effect, so only a structural zero fails.
        reason: Why a $0 here means a coverage hole (the mechanism).
        issue: Tracking issue for the restoration work.
    """

    id: str
    name: str
    parameter_changes: Mapping[str, Any]
    budget_measure: str
    binding_inputs: tuple[str, ...]
    min_abs_effect: float
    reason: str
    issue: str
    effect_direction: str = "reform_minus_baseline"
    period: int | None = None
    expected_sign: str = "positive"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ReformCoverageProbe.id is required.")
        if not self.parameter_changes:
            raise ValueError(f"{self.id}: parameter_changes is required.")
        if not self.budget_measure:
            raise ValueError(f"{self.id}: budget_measure is required.")
        if self.effect_direction not in {
            "reform_minus_baseline",
            "baseline_minus_reform",
        }:
            raise ValueError(
                f"{self.id}: effect_direction must be 'reform_minus_baseline' or "
                "'baseline_minus_reform'."
            )
        if not (self.min_abs_effect > 0):
            raise ValueError(f"{self.id}: min_abs_effect must be positive.")
        if self.period is not None and self.period < 1990:
            raise ValueError(f"{self.id}: period must be a plausible year.")
        if self.expected_sign not in {"positive", "negative"}:
            raise ValueError(
                f"{self.id}: expected_sign must be 'positive' or 'negative'."
            )


@dataclass(frozen=True)
class ReleaseInputCoverageManifest:
    """The full parsed coverage contract.

    Attributes:
        reference: Provenance of the reference eCPS the surface is derived from.
        columns: Every declared column, in name order.
        probes: The pinned reform-coverage probes.
        schema_version: Manifest schema version.
    """

    reference: Mapping[str, str]
    columns: tuple[ReleaseInputColumn, ...]
    probes: tuple[ReformCoverageProbe, ...] = ()
    schema_version: int = 1
    _by_name: dict[str, ReleaseInputColumn] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        by_name: dict[str, ReleaseInputColumn] = {}
        for column in self.columns:
            if column.name in by_name:
                raise ValueError(f"Duplicate manifest column {column.name!r}.")
            by_name[column.name] = column
        object.__setattr__(self, "_by_name", by_name)

    @property
    def declared_columns(self) -> frozenset[str]:
        """Every column the manifest declares (required + reviewed)."""
        return frozenset(self._by_name)

    @property
    def required_columns(self) -> frozenset[str]:
        """Columns that must be present and non-degenerate."""
        return frozenset(
            column.name for column in self.columns if column.status == REQUIRED_STATUS
        )

    @property
    def reviewed_exclusions(self) -> dict[str, str]:
        """Reviewed-exclusion columns as ``name -> reason``."""
        return {
            column.name: column.reason
            for column in self.columns
            if column.status == REVIEWED_EXCLUSION_STATUS
        }


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


def load_release_input_coverage_manifest(
    resource: str = US_RELEASE_INPUT_COVERAGE_RESOURCE,
) -> ReleaseInputCoverageManifest:
    """Load and validate the release input-column coverage manifest.

    Raises:
        ValueError: If the payload shape is wrong, a column has an unknown
            status, a reviewed exclusion is missing its reason or issue, or the
            declared column set is empty (a silently-empty manifest would make
            the coverage gate vacuous).
    """
    payload = _resource_payload(resource)

    raw_reference = payload.get("reference")
    if not isinstance(raw_reference, Mapping):
        raise ValueError(f"{resource}: 'reference' must be a JSON object.")
    reference = {str(key): str(value) for key, value in raw_reference.items()}

    raw_columns = payload.get("columns")
    if not isinstance(raw_columns, Mapping) or not raw_columns:
        raise ValueError(
            f"{resource}: 'columns' must be a non-empty JSON object; a silently "
            "empty manifest would make the coverage gate vacuous."
        )
    columns: list[ReleaseInputColumn] = []
    for name, entry in sorted(raw_columns.items()):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{resource}: column {name!r} must be a JSON object.")
        columns.append(
            ReleaseInputColumn(
                name=str(name),
                status=str(entry.get("status", "")),
                reason=str(entry.get("reason", "")),
                issue=str(entry.get("issue", "")),
                note=str(entry.get("note", "")),
            )
        )

    probes: list[ReformCoverageProbe] = []
    for raw_probe in payload.get("reform_coverage_probes", []) or []:
        if not isinstance(raw_probe, Mapping):
            raise ValueError(f"{resource}: each reform probe must be a JSON object.")
        parameter_changes = raw_probe.get("parameter_changes")
        if not isinstance(parameter_changes, Mapping):
            raise ValueError(
                f"{resource}: probe {raw_probe.get('id')!r} parameter_changes "
                "must be a JSON object."
            )
        probes.append(
            ReformCoverageProbe(
                id=str(raw_probe.get("id", "")),
                name=str(raw_probe.get("name", "")),
                parameter_changes=dict(parameter_changes),
                budget_measure=str(raw_probe.get("budget_measure", "")),
                binding_inputs=tuple(
                    str(leaf) for leaf in raw_probe.get("binding_inputs", ())
                ),
                min_abs_effect=float(raw_probe.get("min_abs_effect", 0.0)),
                reason=str(raw_probe.get("reason", "")),
                issue=str(raw_probe.get("issue", "")),
                effect_direction=str(
                    raw_probe.get("effect_direction", "reform_minus_baseline")
                ),
                period=(
                    None
                    if raw_probe.get("period") is None
                    else int(raw_probe["period"])
                ),
                expected_sign=str(raw_probe.get("expected_sign", "positive")),
            )
        )

    schema_version = payload.get("schema_version", 1)
    if not isinstance(schema_version, int):
        raise ValueError(f"{resource}: 'schema_version' must be an integer.")

    return ReleaseInputCoverageManifest(
        reference=reference,
        columns=tuple(columns),
        probes=tuple(probes),
        schema_version=schema_version,
    )


def us_release_input_coverage_required_columns() -> frozenset[str]:
    """The hard-required input columns from the shipped manifest."""
    return load_release_input_coverage_manifest().required_columns


def us_release_input_coverage_reviewed_exclusions() -> dict[str, str]:
    """The reviewed-exclusion register from the shipped manifest."""
    return load_release_input_coverage_manifest().reviewed_exclusions


def us_release_reform_coverage_probes() -> tuple[ReformCoverageProbe, ...]:
    """The pinned reform-coverage probes from the shipped manifest."""
    return load_release_input_coverage_manifest().probes


def _degenerate_columns(
    column_values: Mapping[str, Any],
    engine: Any,
) -> set[str]:
    """Columns whose every observed value equals the engine default.

    Reuses :func:`populace.build.gates.default_valued_columns_gate`'s exact
    degeneracy classification (same scalar/enum comparison the degenerate-input
    gate uses) so "present but default" is judged identically everywhere.
    """
    from populace.build.gates import default_valued_columns_gate

    defaults = engine.default_values(sorted(column_values))
    classification = default_valued_columns_gate(column_values, defaults)
    return set(classification.details["default_valued_columns"])


def us_release_input_coverage_gate(
    frame: Any,
    engine: Any,
    *,
    manifest: ReleaseInputCoverageManifest | None = None,
) -> GateResult:
    """Build the named US release input-column coverage gate for an export frame.

    Every ``required`` manifest column must be persisted by ``frame`` as a key
    with at least one non-default value; a required column that is absent or
    degenerate (every value the engine default) fails the gate, and a reviewed
    exclusion whose column now carries signal is stale and fails too (#286
    cannot-rot). Run on the calibrated export frame just before
    ``write_dataset``, it hard-fails the release like the export-mass parity gate.

    Args:
        frame: The export :class:`populace.frame.Frame`.
        engine: A ``PolicyEngineUSEngine`` (for input-variable defaults).
        manifest: Override the shipped manifest (tests).

    Returns:
        The ``"us_release_input_coverage"`` gate result.
    """
    manifest = manifest or load_release_input_coverage_manifest()
    required = manifest.required_columns
    reviewed = manifest.reviewed_exclusions
    relevant = required | set(reviewed)

    present_values: dict[str, Any] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        for column in table.columns:
            if column in relevant and column not in present_values:
                present_values[column] = table[column].to_numpy()

    degenerate = _degenerate_columns(present_values, engine)
    return input_column_coverage_gate(
        present_values.keys(),
        required_columns=required,
        degenerate_columns=degenerate,
        reviewed_exclusions=reviewed,
        name="us_release_input_coverage",
    )


def _coverage_engine() -> Any | None:
    """A PolicyEngine-US adapter for graph checks, or ``None`` if absent.

    Mirrors the validation-leaf registry's lazy engine: succeeds without the
    ``[us]`` extra and treats a missing extra at call time as ``None``, so the
    manifest consistency check is a no-op off-build and runs live where the
    extra is installed (the build).
    """
    try:
        from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine
    except ImportError:
        return None
    try:
        return PolicyEngineUSEngine()
    except Exception:  # pragma: no cover - defensive
        return None


def _ecps_populated_layers() -> frozenset[str]:
    payload = _resource_payload(_ECPS_PARITY_REFERENCE_RESOURCE)
    shares = payload.get("nonzero_shares")
    if not isinstance(shares, Mapping) or not shares:
        raise ValueError(
            f"{_ECPS_PARITY_REFERENCE_RESOURCE}: 'nonzero_shares' must be a "
            "non-empty JSON object."
        )
    return frozenset(str(name) for name, share in shares.items() if float(share) > 0.0)


def assert_release_input_coverage_manifest_current(
    *,
    engine: Any | None = None,
    manifest: ReleaseInputCoverageManifest | None = None,
) -> None:
    """Fail if the coverage manifest has drifted from its authoritative sources.

    Checked against the checked-in facts (always) and the live engine graph
    (when available):

    - The declared columns must equal the reference eCPS populated input surface
      (``ecps_parity_reference.json``): the manifest is the eCPS export surface,
      no more, no less. A layer the incumbent gains/loses must be reflected here.
    - The three SSI countable-resource asset inputs must be ``required`` with no
      reviewed exclusion — the #368 red-gate guarantee cannot be quietly undone.
    - Every declared column must be a real PolicyEngine-US input leaf, and every
      probe's ``binding_inputs`` / ``budget_measure`` must resolve on the live
      engine, so the contract cannot guard names the engine no longer has.

    A no-op for the engine-graph half when no engine is available (the workspace
    test environment); the checked-in-facts half always runs.

    Raises:
        ValueError: Naming every drift found.
    """
    manifest = manifest or load_release_input_coverage_manifest()
    failures: list[str] = []

    declared = set(manifest.declared_columns)
    surface = set(_ecps_populated_layers())
    missing_from_manifest = sorted(surface - declared)
    extra_in_manifest = sorted(declared - surface)
    if missing_from_manifest:
        failures.append(
            "manifest is missing reference eCPS populated input column(s) "
            f"{missing_from_manifest}; regenerate with "
            "tools/build_us_release_input_coverage_manifest.py."
        )
    if extra_in_manifest:
        failures.append(
            "manifest declares column(s) the reference eCPS does not populate "
            f"{extra_in_manifest}; regenerate the manifest."
        )

    required = manifest.required_columns
    reviewed = set(manifest.reviewed_exclusions)
    for asset in SSI_COUNTABLE_RESOURCE_ASSETS:
        if asset in reviewed:
            failures.append(
                f"{asset}: SSI countable-resource asset input is a reviewed "
                "exclusion, but #368 requires it be a hard requirement with no "
                "exclusion so the gate fails until the asset stage is restored."
            )
        elif asset not in required:
            failures.append(
                f"{asset}: SSI countable-resource asset input must be a "
                "required manifest column (#368)."
            )

    if engine is None:
        engine = _coverage_engine()
    if engine is not None:
        try:
            input_variables = set(engine.variables())
        except ImportError:
            input_variables = None
        if input_variables is not None:
            non_leaves = sorted(declared - input_variables)
            if non_leaves:
                failures.append(
                    "manifest declares column(s) that are not PolicyEngine-US "
                    f"input leaves (formula-owned or unknown): {non_leaves}."
                )
            for probe in manifest.probes:
                bad_inputs = sorted(set(probe.binding_inputs) - input_variables)
                if bad_inputs:
                    failures.append(
                        f"probe {probe.id!r}: binding_inputs are not "
                        f"PolicyEngine-US input leaves: {bad_inputs}."
                    )

    if failures:
        raise ValueError(
            "US release input-column coverage manifest has drifted:\n"
            + "\n".join(f"  - {line}" for line in failures)
        )
