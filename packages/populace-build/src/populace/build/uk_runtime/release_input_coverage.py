"""UK release input-column coverage: the enhanced-FRS surface as a hard gate.

Every populated effective loader input extracted from the immutable enhanced
FRS reference is declared in ``release_input_coverage_manifest.json`` as either
``required`` or ``reviewed_exclusion``. Required columns must be present on the
final export tables and carry at least one value different from the live
PolicyEngine-UK default. A reviewed exclusion that later carries signal is
stale and fails, so the debt ledger can only shrink deliberately.

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


def _entity_tables(frame: Any) -> tuple[Any, ...]:
    if hasattr(frame, "entities") and callable(getattr(frame, "table", None)):
        return tuple(frame.table(entity) for entity in frame.entities)
    if isinstance(frame, Mapping):
        return tuple(frame[entity] for entity in ("person", "benunit", "household"))
    tables = tuple(
        getattr(frame, entity, None) for entity in ("person", "benunit", "household")
    )
    if any(table is None for table in tables):
        raise TypeError(
            "UK coverage gate expects a populace Frame, an entity-table mapping, "
            "or an object with person/benunit/household tables."
        )
    return tables


def _degenerate_columns(column_values: Mapping[str, Any], engine: Any) -> set[str]:
    defaults = engine.default_values(sorted(column_values))
    missing_defaults = sorted(set(column_values) - set(defaults))
    if missing_defaults:
        raise ValueError(
            "Cannot enforce UK input coverage without live engine defaults for "
            f"{missing_defaults}."
        )
    return {
        name
        for name, values in column_values.items()
        if not _has_nondefault_signal(values, defaults[name])
    }


def _has_nondefault_signal(values: Any, default: object) -> bool:
    """Whether a column has a valid value that differs from its live default."""
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
        return bool(np.any(np.isfinite(numeric) & (numeric != float(bool(default)))))

    if isinstance(default, int | float | np.integer | np.floating):
        numeric = pd.to_numeric(pd.Series(observed), errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        return bool(np.any(np.isfinite(numeric) & (numeric != float(default))))

    if isinstance(default, str):
        normalized = pd.Series(observed, dtype="string").str.strip()
        signal = (
            normalized.notna() & normalized.ne("") & normalized.ne(default.strip())
        ).fillna(False)
        return bool(signal.any())

    for value in observed:
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
                return True
        except (TypeError, ValueError):
            return True
    return False


def uk_release_input_coverage_gate(
    frame: Any,
    engine: Any,
    *,
    manifest: UKReleaseInputCoverageManifest | None = None,
) -> GateResult:
    """Enforce required presence/signal and cannot-rot UK exclusions."""
    manifest = manifest or load_uk_release_input_coverage_manifest()
    required = manifest.required_columns
    reviewed = manifest.reviewed_exclusions
    relevant = required | set(reviewed)
    present_values: dict[str, Any] = {}
    for table in _entity_tables(frame):
        for column in table.columns:
            if column in relevant and column not in present_values:
                present_values[column] = table[column].to_numpy()
    degenerate = _degenerate_columns(present_values, engine)
    return input_column_coverage_gate(
        present_values.keys(),
        required_columns=required,
        degenerate_columns=degenerate,
        reviewed_exclusions=reviewed,
        name="uk_release_input_coverage",
        reference_label="enhanced-FRS",
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
