"""UK release input-column coverage: the enhanced-FRS surface as a hard gate.

Every populated effective loader input extracted from the immutable enhanced
FRS reference is declared in ``release_input_coverage_manifest.json`` as either
``required`` or ``reviewed_exclusion``. Required columns must be present on the
final export tables and carry non-default signal on rows with enough effective
population mass to clear the reviewed manifest floor. A reviewed exclusion
that later carries effective signal is stale and fails, so the debt ledger can
only shrink deliberately.

Three persisted compatibility columns are real UK loader inputs even though
their source variables are formula-owned: ``Simulation.__init__`` moves them to
canonical leaves after loading. They remain in the contract through
``UK_LOADER_INPUT_ALIASES``; this prevents a mechanically strict formula filter
from dropping employment income, capital gains, and employee pension
contributions from the coverage boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import GateResult, input_column_coverage_gate
from populace.build.uk_runtime.parity_reference import load_efrs_parity_known_gaps

__all__ = [
    "UKEffectiveMassCoveragePolicy",
    "UK_LOADER_INPUT_ALIASES",
    "UK_RELEASE_INPUT_COVERAGE_RESOURCE",
    "PolicyEngineUKCoverageEngine",
    "UKReleaseInputColumn",
    "UKReleaseInputCoverageManifest",
    "assert_uk_release_input_coverage_manifest_current",
    "load_uk_release_input_coverage_manifest",
    "uk_release_input_coverage_gate",
    "uk_release_input_coverage_required_columns",
    "uk_release_input_coverage_reviewed_exclusions",
]

UK_RELEASE_INPUT_COVERAGE_RESOURCE = "release_input_coverage_manifest.json"
UK_LOADER_INPUT_ALIASES: Mapping[str, str] = {
    "capital_gains": "capital_gains_before_response",
    "employee_pension_contributions": ("employee_pension_contributions_reported"),
    "employment_income": "employment_income_before_lsr",
}

REQUIRED_STATUS = "required"
REVIEWED_EXCLUSION_STATUS = "reviewed_exclusion"
_VALID_STATUSES = frozenset({REQUIRED_STATUS, REVIEWED_EXCLUSION_STATUS})
_UK_PACKAGE = "populace.build.uk"
_EFRS_PARITY_REFERENCE_RESOURCE = "efrs_parity_reference.json"

DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE = 1e-6


@dataclass(frozen=True)
class UKEffectiveMassCoveragePolicy:
    """Reviewed threshold for signal carried by real population mass."""

    minimum_nondefault_mass_share: float = DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE
    weight_source: str = "household_weight"
    reviewed_on: str = "2026-07-11"
    rationale: str = (
        "One part per million rejects zero-weight support and numerical dust "
        "while remaining about 100 times below the rarest populated record "
        "share in the pinned enhanced-FRS reference."
    )

    def __post_init__(self) -> None:
        floor = float(self.minimum_nondefault_mass_share)
        if not np.isfinite(floor) or not 0.0 < floor <= 1.0:
            raise ValueError(
                "minimum_nondefault_mass_share must be finite and in (0, 1]."
            )
        if self.weight_source != "household_weight":
            raise ValueError(
                "UK effective-mass coverage must use household_weight, the "
                "calibrated population-mass source."
            )
        if not self.reviewed_on.strip():
            raise ValueError("UK effective-mass coverage needs a review date.")
        if not self.rationale.strip():
            raise ValueError("UK effective-mass coverage needs a rationale.")
        object.__setattr__(self, "minimum_nondefault_mass_share", floor)


@dataclass(frozen=True)
class UKReleaseInputColumn:
    """One required input or honestly reasoned, ledger-tracked exclusion."""

    name: str
    status: str
    reason: str = ""
    tracking_note: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("UKReleaseInputColumn.name is required.")
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"{self.name}: status must be one of {sorted(_VALID_STATUSES)}, "
                f"got {self.status!r}."
            )
        if self.status == REVIEWED_EXCLUSION_STATUS:
            if not self.reason:
                raise ValueError(f"{self.name}: a reviewed exclusion needs a reason.")
            if not self.tracking_note:
                raise ValueError(
                    f"{self.name}: a reviewed exclusion needs a tracking note."
                )


@dataclass(frozen=True)
class UKReleaseInputCoverageManifest:
    """Parsed UK input-coverage contract."""

    reference: Mapping[str, str]
    candidate_evidence: Mapping[str, str]
    columns: tuple[UKReleaseInputColumn, ...]
    effective_mass_coverage: UKEffectiveMassCoveragePolicy = field(
        default_factory=UKEffectiveMassCoveragePolicy
    )
    schema_version: int = 1
    _by_name: dict[str, UKReleaseInputColumn] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        by_name: dict[str, UKReleaseInputColumn] = {}
        for column in self.columns:
            if column.name in by_name:
                raise ValueError(f"Duplicate UK manifest column {column.name!r}.")
            by_name[column.name] = column
        object.__setattr__(self, "_by_name", by_name)

    @property
    def declared_columns(self) -> frozenset[str]:
        return frozenset(self._by_name)

    @property
    def required_columns(self) -> frozenset[str]:
        return frozenset(
            column.name for column in self.columns if column.status == REQUIRED_STATUS
        )

    @property
    def reviewed_exclusions(self) -> dict[str, str]:
        return {
            column.name: column.reason
            for column in self.columns
            if column.status == REVIEWED_EXCLUSION_STATUS
        }


def _resource_text(resource: str) -> str:
    candidate = Path(resource)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return files(_UK_PACKAGE).joinpath(resource).read_text(encoding="utf-8")


def _resource_payload(resource: str) -> Mapping[str, Any]:
    payload = json.loads(_resource_text(resource))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resource}: expected a JSON object.")
    return payload


def load_uk_release_input_coverage_manifest(
    resource: str = UK_RELEASE_INPUT_COVERAGE_RESOURCE,
) -> UKReleaseInputCoverageManifest:
    """Load the non-vacuous UK manifest and validate exclusion metadata."""
    payload = _resource_payload(resource)
    raw_reference = payload.get("reference")
    raw_candidate = payload.get("candidate_evidence")
    if not isinstance(raw_reference, Mapping):
        raise ValueError(f"{resource}: 'reference' must be a JSON object.")
    if not isinstance(raw_candidate, Mapping):
        raise ValueError(f"{resource}: 'candidate_evidence' must be a JSON object.")

    raw_effective_mass = payload.get("effective_mass_coverage")
    if not isinstance(raw_effective_mass, Mapping):
        raise ValueError(
            f"{resource}: 'effective_mass_coverage' must be a JSON object."
        )
    effective_mass_coverage = UKEffectiveMassCoveragePolicy(
        minimum_nondefault_mass_share=float(
            raw_effective_mass.get("minimum_nondefault_mass_share", 0.0)
        ),
        weight_source=str(raw_effective_mass.get("weight_source", "")),
        reviewed_on=str(raw_effective_mass.get("reviewed_on", "")),
        rationale=str(raw_effective_mass.get("rationale", "")),
    )

    raw_columns = payload.get("columns")
    if not isinstance(raw_columns, Mapping) or not raw_columns:
        raise ValueError(
            f"{resource}: 'columns' must be a non-empty JSON object; a silently "
            "empty manifest would make the coverage gate vacuous."
        )
    columns: list[UKReleaseInputColumn] = []
    for name, entry in sorted(raw_columns.items()):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{resource}: column {name!r} must be a JSON object.")
        columns.append(
            UKReleaseInputColumn(
                name=str(name),
                status=str(entry.get("status", "")),
                reason=str(entry.get("reason", "")),
                tracking_note=str(entry.get("tracking_note", "")),
                note=str(entry.get("note", "")),
            )
        )
    schema_version = payload.get("schema_version", 1)
    if not isinstance(schema_version, int):
        raise ValueError(f"{resource}: 'schema_version' must be an integer.")
    return UKReleaseInputCoverageManifest(
        reference={str(key): str(value) for key, value in raw_reference.items()},
        candidate_evidence={
            str(key): str(value) for key, value in raw_candidate.items()
        },
        columns=tuple(columns),
        effective_mass_coverage=effective_mass_coverage,
        schema_version=schema_version,
    )


def uk_release_input_coverage_required_columns() -> frozenset[str]:
    return load_uk_release_input_coverage_manifest().required_columns


def uk_release_input_coverage_reviewed_exclusions() -> dict[str, str]:
    return load_uk_release_input_coverage_manifest().reviewed_exclusions


def _stored_enum_name(value: object) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else str(value)


class PolicyEngineUKCoverageEngine:
    """Small lazy adapter exposing UK effective inputs and stored defaults."""

    def __init__(self) -> None:
        self._system: Any | None = None

    def _tax_benefit_system(self) -> Any:
        if self._system is None:
            from policyengine_uk import CountryTaxBenefitSystem

            self._system = CountryTaxBenefitSystem()
        return self._system

    def variables(self) -> list[str]:
        variables = self._tax_benefit_system().variables
        pure_inputs = {
            name for name, variable in variables.items() if variable.is_input_variable()
        }
        aliases: set[str] = set()
        for source, target in UK_LOADER_INPUT_ALIASES.items():
            if source not in variables:
                raise ValueError(f"UK loader alias source {source!r} is unknown.")
            if target not in pure_inputs:
                raise ValueError(
                    f"UK loader alias target {target!r} is no longer an input leaf."
                )
            aliases.add(source)
        return sorted(pure_inputs | aliases)

    def default_values(self, names: Sequence[str]) -> dict[str, object]:
        variables = self._tax_benefit_system().variables
        effective = set(self.variables())
        defaults: dict[str, object] = {}
        for name in names:
            if name not in effective:
                continue
            variable = variables.get(name)
            if variable is None:
                continue
            default = getattr(variable, "default_value", None)
            if default is None:
                continue
            value_type = getattr(variable, "value_type", None)
            if value_type in {bool, int, float, str}:
                defaults[name] = default
            else:
                stored = _stored_enum_name(default)
                if stored is not None:
                    defaults[name] = stored
        return defaults


def _entity_tables(frame: Any) -> tuple[tuple[str, Any], ...]:
    if hasattr(frame, "entities") and callable(getattr(frame, "table", None)):
        return tuple((entity, frame.table(entity)) for entity in frame.entities)
    if isinstance(frame, Mapping):
        return tuple(
            (entity, frame[entity]) for entity in ("person", "benunit", "household")
        )
    tables = tuple(
        (entity, getattr(frame, entity, None))
        for entity in ("person", "benunit", "household")
    )
    if any(table is None for _entity, table in tables):
        raise TypeError(
            "UK coverage gate expects a populace Frame, an entity-table mapping, "
            "or an object with person/benunit/household tables."
        )
    return tables


def _nondefault_signal_mask(values: Any, default: object) -> np.ndarray:
    """Row mask for valid values that differ from a live engine default."""
    observed = np.asarray(values)
    if observed.ndim == 0:
        observed = observed.reshape(1)
    else:
        observed = observed.reshape(-1)

    if isinstance(default, bool | np.bool_):
        if observed.dtype.kind in {"b", "i", "u", "f"}:
            numeric = observed.astype(float, copy=False)
        else:
            normalized = pd.Series(observed, dtype="string").str.strip().str.lower()
            numeric = normalized.map(
                {"false": 0.0, "0": 0.0, "true": 1.0, "1": 1.0}
            ).to_numpy(dtype=float, na_value=np.nan)
        return np.isfinite(numeric) & (numeric != float(bool(default)))

    if isinstance(default, int | float | np.integer | np.floating):
        numeric = pd.to_numeric(pd.Series(observed), errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        return np.isfinite(numeric) & (numeric != float(default))

    if isinstance(default, str):
        normalized = pd.Series(observed, dtype="string").str.strip()
        signal = (
            normalized.notna() & normalized.ne("") & normalized.ne(default.strip())
        ).fillna(False)
        return signal.to_numpy(dtype=bool)

    signal = np.zeros(len(observed), dtype=bool)
    for index, value in enumerate(observed):
        missing = pd.isna(value)
        if isinstance(missing, bool | np.bool_) and missing:
            continue
        if isinstance(value, bytes):
            if not value.decode(errors="replace").strip():
                continue
        elif isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, float | np.floating) and not np.isfinite(value):
            continue
        try:
            if value != default:
                signal[index] = True
        except (TypeError, ValueError):
            signal[index] = True
    return signal


def _effective_weights_by_entity(
    frame: Any,
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    """Resolve row-aligned effective population mass for all UK entities."""
    resolve_weights = getattr(frame, "resolve_weights", None)
    if callable(resolve_weights):
        return {
            entity: np.asarray(resolve_weights(entity).values, dtype=np.float64)
            for entity in tables
        }

    household = tables["household"]
    person = tables["person"]
    if "household_weight" not in household:
        raise ValueError(
            "UK effective-mass coverage requires household.household_weight."
        )
    household_weights = pd.to_numeric(
        household["household_weight"], errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    if (
        not np.isfinite(household_weights).all()
        or (household_weights < 0.0).any()
        or not (household_weights > 0.0).any()
    ):
        raise ValueError(
            "UK effective-mass coverage requires finite, non-negative "
            "household weights with positive total mass."
        )
    household_ids = household["household_id"]
    if household_ids.isna().any() or household_ids.duplicated().any():
        raise ValueError(
            "UK effective-mass coverage requires unique household_id values."
        )
    weights_by_household = pd.Series(
        household_weights,
        index=household_ids.to_numpy(),
    )
    person_weights = person["person_household_id"].map(weights_by_household)
    if person_weights.isna().any():
        raise ValueError(
            "UK effective-mass coverage cannot map every person to a weighted "
            "household."
        )

    benunit_households = (
        person[["person_benunit_id", "person_household_id"]]
        .drop_duplicates()
        .groupby("person_benunit_id", sort=False)["person_household_id"]
        .agg(list)
    )
    ambiguous = benunit_households[benunit_households.map(len) != 1]
    if len(ambiguous):
        raise ValueError(
            "UK effective-mass coverage requires each benunit to belong to "
            "exactly one household."
        )
    household_by_benunit = benunit_households.map(lambda values: values[0])
    benunit_weights = tables["benunit"]["benunit_id"].map(
        household_by_benunit.map(weights_by_household)
    )
    if benunit_weights.isna().any():
        raise ValueError(
            "UK effective-mass coverage cannot map every benunit to a weighted "
            "household."
        )
    return {
        "person": person_weights.to_numpy(dtype=np.float64),
        "benunit": benunit_weights.to_numpy(dtype=np.float64),
        "household": household_weights,
    }


def _effective_mass_diagnostics(
    column_values: Mapping[str, tuple[str, Any]],
    *,
    defaults: Mapping[str, object],
    effective_weights: Mapping[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    diagnostics: dict[str, dict[str, object]] = {}
    for name, (entity, values) in sorted(column_values.items()):
        weights = np.asarray(effective_weights[entity], dtype=np.float64)
        signal = _nondefault_signal_mask(values, defaults[name])
        if len(signal) != len(weights):
            raise ValueError(
                f"{name}: column length {len(signal)} disagrees with {entity} "
                f"effective-weight length {len(weights)}."
            )
        positive = weights > 0.0
        effective_signal = signal & positive
        total_mass = float(weights[positive].sum())
        signal_mass = float(weights[effective_signal].sum())
        share = signal_mass / total_mass if total_mass > 0.0 else 0.0
        diagnostics[name] = {
            "entity": entity,
            "signal_rows": int(signal.sum()),
            "positive_mass_signal_rows": int(effective_signal.sum()),
            "effective_signal_mass": signal_mass,
            "total_effective_mass": total_mass,
            "effective_signal_mass_share": share,
        }
    return diagnostics


def uk_release_input_coverage_gate(
    frame: Any,
    engine: Any,
    *,
    manifest: UKReleaseInputCoverageManifest | None = None,
) -> GateResult:
    """Enforce required signal on rows carrying reviewed effective mass."""
    manifest = manifest or load_uk_release_input_coverage_manifest()
    required = manifest.required_columns
    reviewed = manifest.reviewed_exclusions
    relevant = required | set(reviewed)
    table_items = _entity_tables(frame)
    tables = {entity: table for entity, table in table_items}
    effective_weights = _effective_weights_by_entity(frame, tables)
    present_values: dict[str, tuple[str, Any]] = {}
    for entity, table in table_items:
        for column in table.columns:
            if column not in relevant:
                continue
            if column in present_values:
                prior_entity = present_values[column][0]
                raise ValueError(
                    f"UK coverage column {column!r} occurs on both "
                    f"{prior_entity!r} and {entity!r} tables."
                )
            present_values[column] = (entity, table[column].to_numpy())

    defaults = engine.default_values(sorted(present_values))
    missing_defaults = sorted(set(present_values) - set(defaults))
    if missing_defaults:
        raise ValueError(
            "Cannot enforce UK input coverage without live engine defaults for "
            f"{missing_defaults}."
        )
    diagnostics = _effective_mass_diagnostics(
        present_values,
        defaults=defaults,
        effective_weights=effective_weights,
    )
    raw_degenerate = {
        name
        for name, (_entity, values) in present_values.items()
        if not _nondefault_signal_mask(values, defaults[name]).any()
    }
    floor = manifest.effective_mass_coverage.minimum_nondefault_mass_share
    insufficient_effective_mass = {
        name
        for name, details in diagnostics.items()
        if float(details["effective_signal_mass_share"]) < floor
    }
    degenerate = raw_degenerate | insufficient_effective_mass
    base = input_column_coverage_gate(
        present_values.keys(),
        required_columns=required,
        degenerate_columns=degenerate,
        reviewed_exclusions=reviewed,
        name="uk_release_input_coverage",
        reference_label="enhanced-FRS",
    )
    failures = list(base.failures)
    for name in sorted((insufficient_effective_mass - raw_degenerate) & required):
        prefix = f"{name}: required enhanced-FRS input column is present but every "
        failures = [failure for failure in failures if not failure.startswith(prefix)]
        column_details = diagnostics[name]
        failures.append(
            f"{name}: required enhanced-FRS input column carries non-default "
            f"signal on effective population mass share "
            f"{float(column_details['effective_signal_mass_share']):.12g}, below "
            f"the reviewed floor {floor:.12g}; zero-weight support or numerical "
            "dust is not release coverage. Rebuild the source channel with a "
            "reviewed positive population-mass allocation."
        )
    details = {
        **dict(base.details),
        "effective_mass_policy": {
            "weight_source": manifest.effective_mass_coverage.weight_source,
            "minimum_nondefault_mass_share": floor,
            "reviewed_on": manifest.effective_mass_coverage.reviewed_on,
            "rationale": manifest.effective_mass_coverage.rationale,
        },
        "insufficient_effective_mass": sorted(insufficient_effective_mass),
        "effective_mass_by_column": diagnostics,
    }
    return GateResult(
        name=base.name,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )


def _coverage_engine() -> PolicyEngineUKCoverageEngine | None:
    try:
        engine = PolicyEngineUKCoverageEngine()
        engine.variables()
    except (ImportError, ModuleNotFoundError):
        return None
    return engine


def _efrs_reference_payload() -> Mapping[str, Any]:
    return _resource_payload(_EFRS_PARITY_REFERENCE_RESOURCE)


def _efrs_populated_layers() -> frozenset[str]:
    shares = _efrs_reference_payload().get("nonzero_shares")
    if not isinstance(shares, Mapping) or not shares:
        raise ValueError(
            f"{_EFRS_PARITY_REFERENCE_RESOURCE}: 'nonzero_shares' must be a "
            "non-empty JSON object."
        )
    return frozenset(str(name) for name, share in shares.items() if float(share) > 0.0)


def assert_uk_release_input_coverage_manifest_current(
    *,
    engine: Any | None = None,
    manifest: UKReleaseInputCoverageManifest | None = None,
) -> None:
    """Fail before a build if the UK contract has narrowed or graph-rotted."""
    manifest = manifest or load_uk_release_input_coverage_manifest()
    failures: list[str] = []
    if manifest.effective_mass_coverage != UKEffectiveMassCoveragePolicy():
        failures.append(
            "effective_mass_coverage disagrees with the reviewed UK runtime "
            "policy; regenerate the manifest or review the code and manifest "
            "together."
        )
    declared = set(manifest.declared_columns)
    surface = set(_efrs_populated_layers())
    missing = sorted(surface - declared)
    extra = sorted(declared - surface)
    if missing:
        failures.append(
            "manifest is missing enhanced-FRS populated loader input(s) "
            f"{missing}; regenerate with "
            "tools/build_uk_release_input_coverage_manifest.py."
        )
    if extra:
        failures.append(
            "manifest declares column(s) outside the enhanced-FRS populated "
            f"loader-input surface: {extra}."
        )

    known_gaps = {gap.variable: gap for gap in load_efrs_parity_known_gaps()}
    gaps_outside_surface = sorted(set(known_gaps) - surface)
    if gaps_outside_surface:
        failures.append(
            "efrs_parity_known_gaps.json declares column(s) outside the "
            f"enhanced-FRS populated surface: {gaps_outside_surface}."
        )
    for name in sorted(surface & declared):
        column = manifest._by_name[name]
        gap = known_gaps.get(name)
        if gap is None:
            if column.status != REQUIRED_STATUS:
                failures.append(
                    f"{name}: efrs_parity_known_gaps.json has no gap, so the "
                    "manifest status must remain required."
                )
            continue
        if column.status != REVIEWED_EXCLUSION_STATUS:
            failures.append(
                f"{name}: committed known gap must be a reviewed_exclusion in "
                "the manifest."
            )
            continue
        if column.reason != gap.reason:
            failures.append(
                f"{name}: manifest exclusion reason disagrees with "
                "efrs_parity_known_gaps.json."
            )
        if column.tracking_note != gap.tracking_note:
            failures.append(
                f"{name}: manifest exclusion tracking note disagrees with "
                "efrs_parity_known_gaps.json."
            )

    raw_engine = _efrs_reference_payload().get("engine")
    raw_aliases = (
        raw_engine.get("h5_input_aliases") if isinstance(raw_engine, Mapping) else None
    )
    if raw_aliases != dict(UK_LOADER_INPUT_ALIASES):
        failures.append(
            "efrs_parity_reference.json h5_input_aliases disagree with the "
            "live UK coverage adapter; regenerate or update both together."
        )

    if engine is None:
        engine = _coverage_engine()
    if engine is not None:
        try:
            input_variables = set(engine.variables())
        except ImportError:
            input_variables = set()
        non_inputs = sorted(declared - input_variables)
        if non_inputs:
            failures.append(
                "manifest declares column(s) that are not live PolicyEngine-UK "
                f"input leaves or loader aliases: {non_inputs}."
            )

    if failures:
        raise ValueError(
            "UK release input-column coverage manifest has drifted:\n"
            + "\n".join(f"  - {line}" for line in failures)
        )
