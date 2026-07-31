"""UK release input-column coverage: the enhanced-FRS surface as a hard gate.

Every populated effective loader input extracted from the immutable enhanced
FRS reference is declared in ``release_input_coverage_manifest.json`` as either
``required`` or ``reviewed_exclusion``. Required columns must be present on the
final export tables and carry non-default signal on rows with enough effective
population mass to clear the reviewed manifest floor. A reviewed exclusion
that later carries effective signal is stale and fails, so the debt ledger can
only shrink deliberately.

The UK loader calls ``set_input`` for every engine-known persisted H5 column,
including formula-owned variables. Those persisted overrides are therefore
part of the input contract alongside pure input leaves. Three compatibility
columns are additionally moved to canonical leaves by ``Simulation.__init__``;
``UK_LOADER_INPUT_ALIASES`` keeps that migration seam explicit and checked.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import GateResult, input_column_coverage_gate
from populace.build.uk_runtime.parity_reference import (
    load_efrs_parity_known_gaps,
    load_efrs_parity_reference,
)
from populace.build.uk_runtime.release_identity import (
    UK_RELEASE_TIER_FRS,
    validate_uk_release_tier,
)

__all__ = [
    "RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS",
    "UKEffectiveMassCoveragePolicy",
    "UK_LOADER_INPUT_ALIASES",
    "UK_RELEASE_INPUT_COVERAGE_RESOURCE",
    "PolicyEngineUKCoverageEngine",
    "UKReleaseInputColumn",
    "UKReleaseInputCoverageManifest",
    "assert_uk_release_input_coverage_build_stages",
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
_REQUIRED_AT_BUILD_STATUS = "required_at_build"
_DEFERRED_UNTIL_RESTORED_STATUS = "deferred_until_restored"
_VALID_FAMILY_STATUSES = frozenset(
    {_REQUIRED_AT_BUILD_STATUS, _DEFERRED_UNTIL_RESTORED_STATUS}
)
_DISTRIBUTIONAL_REQUIRED_STATUS = "distributional_required"
_UK_PACKAGE = "populace.build.uk"
_EFRS_PARITY_REFERENCE_RESOURCE = "efrs_parity_reference.json"

DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE = 1e-6

# Reference-populated inputs restored after the certified base candidate. This
# runtime copy mirrors the generator's reviewed registry so manifest drift can
# never put a shipped restoration back behind an exclusion.
RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS = frozenset(
    {"charitable_investment_gifts", "gift_aid"}
)


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
    family_coverage: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
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

    @property
    def required_build_stages(self) -> frozenset[str]:
        """National stages that the checked-in family contract makes mandatory."""

        return frozenset(
            str(family["stage"])
            for family in self.family_coverage.values()
            if family.get("status") == _REQUIRED_AT_BUILD_STATUS
        )


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


def _source_stage_base_candidate_tier(
    source_manifest: str,
    *,
    stage_name: str,
) -> str:
    payload = _resource_payload(source_manifest)
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"{source_manifest}: 'stages' must be a JSON list.")
    matching = [
        stage
        for stage in stages
        if isinstance(stage, Mapping) and stage.get("stage") == stage_name
    ]
    if len(matching) != 1:
        raise ValueError(
            f"{source_manifest}: expected exactly one {stage_name!r} stage."
        )
    base_candidate = matching[0].get("base_candidate")
    if not isinstance(base_candidate, Mapping):
        raise ValueError(
            f"{source_manifest}: stage {stage_name!r} needs base_candidate."
        )
    return validate_uk_release_tier(base_candidate.get("tier"))


def _parse_family_coverage(
    raw: object,
    *,
    resource: str,
) -> dict[str, dict[str, Any]]:
    """Validate executable and explicitly deferred family coverage metadata."""

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{resource}: 'family_coverage' must be a JSON object.")

    families: dict[str, dict[str, Any]] = {}
    seen_stages: set[str] = set()
    for raw_name, raw_family in sorted(raw.items()):
        name = str(raw_name)
        if not name or not isinstance(raw_family, Mapping):
            raise ValueError(
                f"{resource}: family coverage entry {raw_name!r} must be an object."
            )
        status = str(raw_family.get("status", ""))
        if status not in _VALID_FAMILY_STATUSES:
            raise ValueError(
                f"{resource}: family {name!r} status must be one of "
                f"{sorted(_VALID_FAMILY_STATUSES)}, got {status!r}."
            )
        restoration_status = str(raw_family.get("restoration_status", "")).strip()
        if status == _DEFERRED_UNTIL_RESTORED_STATUS and not restoration_status:
            raise ValueError(
                f"{resource}: deferred family {name!r} needs a "
                "restoration_status explaining its blocker."
            )
        stage = str(raw_family.get("stage", "")).strip()
        if not stage:
            raise ValueError(f"{resource}: family {name!r} needs a build stage.")
        if stage in seen_stages:
            raise ValueError(
                f"{resource}: required build stage {stage!r} is declared twice."
            )
        seen_stages.add(stage)
        source_manifest = str(raw_family.get("source_manifest", "")).strip()
        source_manifest_sha256 = str(
            raw_family.get("source_manifest_sha256", "")
        ).strip()
        if not source_manifest:
            raise ValueError(f"{resource}: family {name!r} needs a source_manifest.")
        if len(source_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_manifest_sha256
        ):
            raise ValueError(
                f"{resource}: family {name!r} needs a lowercase SHA-256 "
                "for source_manifest_sha256."
            )
        try:
            base_candidate_tier = validate_uk_release_tier(
                raw_family.get("base_candidate_tier")
            )
        except ValueError as exc:
            raise ValueError(
                f"{resource}: family {name!r} has invalid base_candidate_tier: {exc}"
            ) from exc
        output_weight_kind = str(raw_family.get("output_weight_kind", "")).strip()
        if not output_weight_kind:
            raise ValueError(
                f"{resource}: family {name!r} needs an output_weight_kind."
            )
        required_mass_change_reason = str(
            raw_family.get("required_mass_change_reason", "")
        ).strip()
        if not required_mass_change_reason:
            raise ValueError(
                f"{resource}: family {name!r} needs a reviewed "
                "required_mass_change_reason."
            )

        raw_requirements = raw_family.get("effective_mass_requirements", {})
        if not isinstance(raw_requirements, Mapping):
            raise ValueError(
                f"{resource}: family {name!r} effective_mass_requirements must "
                "be an object."
            )
        requirements: dict[str, dict[str, Any]] = {}
        for raw_column, raw_requirement in sorted(raw_requirements.items()):
            column = str(raw_column)
            if not column or not isinstance(raw_requirement, Mapping):
                raise ValueError(
                    f"{resource}: family {name!r} effective-mass requirement "
                    f"{raw_column!r} must be an object."
                )
            requirement_status = str(raw_requirement.get("status", ""))
            if requirement_status != _DISTRIBUTIONAL_REQUIRED_STATUS:
                raise ValueError(
                    f"{resource}: family {name!r} column {column!r} must use "
                    f"status {_DISTRIBUTIONAL_REQUIRED_STATUS!r}."
                )
            floor = float(raw_requirement.get("minimum_nondefault_mass_share", 0.0))
            if not np.isfinite(floor) or not 0.0 < floor <= 1.0:
                raise ValueError(
                    f"{resource}: family {name!r} column {column!r} has invalid "
                    "minimum_nondefault_mass_share."
                )
            support_column = str(
                raw_requirement.get("support_channel_column", "")
            ).strip()
            required_channel = str(
                raw_requirement.get("required_support_channel", "")
            ).strip()
            denominator = str(raw_requirement.get("mass_share_denominator", "")).strip()
            if not support_column or not required_channel:
                raise ValueError(
                    f"{resource}: family {name!r} column {column!r} must declare "
                    "both support_channel_column and required_support_channel."
                )
            if denominator != "all_person_effective_mass":
                raise ValueError(
                    f"{resource}: family {name!r} column {column!r} must use "
                    "mass_share_denominator='all_person_effective_mass'."
                )
            requirements[column] = {
                **dict(raw_requirement),
                "status": requirement_status,
                "minimum_nondefault_mass_share": floor,
                "support_channel_column": support_column,
                "required_support_channel": required_channel,
                "mass_share_denominator": denominator,
            }

        families[name] = {
            **dict(raw_family),
            "status": status,
            "restoration_status": restoration_status,
            "stage": stage,
            "source_manifest": source_manifest,
            "source_manifest_sha256": source_manifest_sha256,
            "base_candidate_tier": base_candidate_tier,
            "output_weight_kind": output_weight_kind,
            "required_mass_change_reason": required_mass_change_reason,
            "effective_mass_requirements": requirements,
        }
    return families


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
    try:
        candidate_tier = validate_uk_release_tier(raw_candidate.get("tier"))
    except ValueError as exc:
        raise ValueError(
            f"{resource}: candidate_evidence.tier is invalid: {exc}"
        ) from exc

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
        }
        | {"tier": candidate_tier},
        columns=tuple(columns),
        family_coverage=_parse_family_coverage(
            payload.get("family_coverage"),
            resource=resource,
        ),
        effective_mass_coverage=effective_mass_coverage,
        schema_version=schema_version,
    )


def uk_release_input_coverage_required_columns() -> frozenset[str]:
    return load_uk_release_input_coverage_manifest().required_columns


def uk_release_input_coverage_reviewed_exclusions() -> dict[str, str]:
    return load_uk_release_input_coverage_manifest().reviewed_exclusions


@dataclass(frozen=True)
class _UKEnumDefault:
    """One PolicyEngine enum default in both H5 name and integer encodings."""

    name: str
    index: int
    value: str


def _stored_enum_default(value: object) -> _UKEnumDefault | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    index = getattr(value, "index", None)
    stored_value = getattr(value, "value", None)
    if not isinstance(name, str) or not isinstance(index, int):
        return None
    return _UKEnumDefault(
        name=name,
        index=index,
        value=str(stored_value) if stored_value is not None else name,
    )


class PolicyEngineUKCoverageEngine:
    """Small lazy adapter exposing UK loadable overrides and stored defaults."""

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
        for source, target in UK_LOADER_INPUT_ALIASES.items():
            if source not in variables:
                raise ValueError(f"UK loader alias source {source!r} is unknown.")
            if target not in pure_inputs:
                raise ValueError(
                    f"UK loader alias target {target!r} is no longer an input leaf."
                )
            source_entity = str(variables[source].entity.key)
            target_entity = str(variables[target].entity.key)
            if source_entity != target_entity:
                raise ValueError(
                    f"UK loader alias {source!r} is owned by {source_entity!r}, "
                    f"but {target!r} is owned by {target_entity!r}."
                )
        # build_from_multi_year_dataset passes every engine-known H5 column to
        # set_input, so formula-owned persisted arrays are effective loader
        # inputs too. Keep the pure-input/alias validation above because the
        # post-load move_values seam must not drift silently.
        return sorted(variables)

    def variable_entities(self, names: Sequence[str]) -> dict[str, str]:
        """Return each effective loader override's owning persisted entity."""

        variables = self._tax_benefit_system().variables
        effective = set(self.variables())
        entities: dict[str, str] = {}
        for name in names:
            if name not in effective:
                continue
            variable = variables.get(name)
            entity = getattr(getattr(variable, "entity", None), "key", None)
            if not isinstance(entity, str) or not entity:
                continue
            entities[name] = entity
        return entities

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
                stored = _stored_enum_default(default)
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


def _engine_variable_entities(
    engine: Any,
    names: Sequence[str] | set[str] | frozenset[str],
) -> dict[str, str]:
    """Resolve and validate live owning entities for a coverage surface."""

    resolver = getattr(engine, "variable_entities", None)
    if not callable(resolver):
        raise ValueError(
            "Cannot enforce UK input coverage without live engine owning-entity "
            "metadata (variable_entities)."
        )
    requested = {str(name) for name in names}
    raw = resolver(sorted(requested))
    if not isinstance(raw, Mapping):
        raise ValueError("UK coverage engine variable_entities must return a mapping.")
    entities = {str(name): str(entity) for name, entity in raw.items()}
    missing = sorted(requested - set(entities))
    invalid = {
        name: entity
        for name, entity in entities.items()
        if name in requested and entity not in {"person", "benunit", "household"}
    }
    if missing or invalid:
        raise ValueError(
            "Cannot enforce UK input coverage with incomplete owning-entity "
            f"metadata: missing={missing}, invalid={invalid}."
        )
    return {name: entities[name] for name in sorted(requested)}


def _nondefault_signal_mask(values: Any, default: object) -> np.ndarray:
    """Row mask for valid values that differ from a live engine default."""
    observed = np.asarray(values)
    if observed.ndim == 0:
        observed = observed.reshape(1)
    else:
        observed = observed.reshape(-1)

    if isinstance(default, _UKEnumDefault):
        series = pd.Series(observed)
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        numeric_values = np.isfinite(numeric)
        signal = numeric_values & (numeric != float(default.index))
        if (~numeric_values).any():
            normalized = series.astype("string").str.strip()
            valid = normalized.notna() & normalized.ne("")
            default_names = {default.name, default.value}
            string_signal = (
                valid & ~normalized.isin(default_names) & ~numeric_values
            ).fillna(False)
            signal |= string_signal.to_numpy(dtype=bool)
        return signal

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

    household = tables["household"]
    person = tables["person"]
    resolve_weights = getattr(frame, "resolve_weights", None)
    if callable(resolve_weights):
        # The reviewed UK policy has exactly one mass source. A Frame may also
        # carry explicit person or benunit weights for a different purpose;
        # resolving those entities independently would let such vectors count
        # signal on a zero-household-mass SPI row.
        household_weights = np.asarray(
            resolve_weights("household").values,
            dtype=np.float64,
        )
        if "household_weight" in household:
            stored_weights = pd.to_numeric(
                household["household_weight"], errors="coerce"
            ).to_numpy(dtype=float, na_value=np.nan)
            if not np.allclose(
                household_weights,
                stored_weights,
                rtol=1e-12,
                atol=0.0,
            ):
                raise ValueError(
                    "UK effective-mass coverage found conflicting typed and "
                    "stored household_weight values."
                )
    else:
        if "household_weight" not in household:
            raise ValueError(
                "UK effective-mass coverage requires household.household_weight."
            )
        household_weights = pd.to_numeric(
            household["household_weight"], errors="coerce"
        ).to_numpy(dtype=float, na_value=np.nan)
    if len(household_weights) != len(household):
        raise ValueError(
            "UK effective-mass coverage household weights do not align to the "
            "household table."
        )
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


def _family_effective_mass_diagnostics(
    manifest: UKReleaseInputCoverageManifest,
    *,
    tables: Mapping[str, pd.DataFrame],
    column_values: Mapping[str, tuple[str, Any]],
    defaults: Mapping[str, object],
    effective_weights: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Enforce distributional families on their declared support channels."""

    diagnostics: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for family_name, family in sorted(manifest.family_coverage.items()):
        if family.get("status") != _REQUIRED_AT_BUILD_STATUS:
            continue
        family_details: dict[str, object] = {}
        requirements = family.get("effective_mass_requirements", {})
        for column, requirement in sorted(requirements.items()):
            if column not in column_values:
                failures.append(
                    f"{family_name}: distributional input {column!r} is absent "
                    "after its required build stage."
                )
                continue
            entity, values = column_values[column]
            support_column = str(requirement.get("support_channel_column", ""))
            required_channel = str(requirement.get("required_support_channel", ""))
            if not support_column or not required_channel:
                failures.append(
                    f"{family_name}: {column!r} has no executable support-"
                    "channel requirement."
                )
                continue
            if requirement.get("mass_share_denominator") != "all_person_effective_mass":
                failures.append(
                    f"{family_name}: {column!r} must use all person effective "
                    "mass as its distributional denominator."
                )
                continue
            table = tables[entity]
            if support_column not in table:
                failures.append(
                    f"{family_name}: required stage {family['stage']!r} did not "
                    f"persist support metadata {support_column!r} for {column!r}."
                )
                continue
            weights = np.asarray(effective_weights[entity], dtype=np.float64)
            signal = _nondefault_signal_mask(values, defaults[column])
            channels = table[support_column].astype("string").str.strip()
            in_channel = (
                channels.eq(required_channel).fillna(False).to_numpy(dtype=bool)
            )
            positive = weights > 0.0
            total_mass = float(weights[positive].sum())
            channel_mass = float(weights[positive & in_channel].sum())
            signal_mass = float(weights[positive & in_channel & signal].sum())
            share = signal_mass / total_mass if total_mass > 0.0 else 0.0
            floor = float(requirement["minimum_nondefault_mass_share"])
            family_details[column] = {
                "entity": entity,
                "support_channel_column": support_column,
                "required_support_channel": required_channel,
                "channel_rows": int(in_channel.sum()),
                "positive_mass_channel_rows": int((positive & in_channel).sum()),
                "channel_effective_mass": channel_mass,
                "effective_signal_mass": signal_mass,
                "total_effective_mass": total_mass,
                "effective_signal_mass_share": share,
                "minimum_nondefault_mass_share": floor,
            }
            if share < floor:
                failures.append(
                    f"{family_name}: {column!r} carries non-default signal on "
                    f"required support channel {required_channel!r} at effective "
                    f"population mass share {share:.12g}, below the reviewed "
                    f"floor {floor:.12g}; base-channel signal does not restore "
                    "the rebuilt SPI family."
                )
        diagnostics[family_name] = family_details
    return diagnostics, failures


def _family_build_state_diagnostics(
    frame: Any,
    manifest: UKReleaseInputCoverageManifest,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Enforce reviewed family weight state, period, and mass-change evidence."""

    diagnostics: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for family_name, family in sorted(manifest.family_coverage.items()):
        if family.get("status") != _REQUIRED_AT_BUILD_STATUS:
            continue
        details: dict[str, object] = {}

        expected_kind = str(family.get("output_weight_kind", "")).strip()
        if expected_kind:
            actual_kind = getattr(frame, "household_weight_kind", None)
            if actual_kind is None:
                resolve_weights = getattr(frame, "resolve_weights", None)
                if callable(resolve_weights):
                    actual_kind = resolve_weights("household").kind
            actual_kind_name = str(getattr(actual_kind, "value", actual_kind or ""))
            details["expected_output_weight_kind"] = expected_kind
            details["actual_output_weight_kind"] = actual_kind_name
            if actual_kind_name != expected_kind:
                failures.append(
                    f"{family_name}: final household weights have kind "
                    f"{actual_kind_name!r}, expected reviewed kind "
                    f"{expected_kind!r}."
                )

        source_vintages = family.get("source_vintages")
        if isinstance(source_vintages, Mapping) and (
            "mapped_build_period" in source_vintages
        ):
            expected_period = str(source_vintages["mapped_build_period"])
            actual_period = str(getattr(frame, "time_period", ""))
            details["expected_build_period"] = expected_period
            details["actual_build_period"] = actual_period
            if actual_period != expected_period:
                failures.append(
                    f"{family_name}: final dataset period {actual_period!r} does "
                    f"not match the reviewed source mapping {expected_period!r}."
                )

        required_reason = str(family.get("required_mass_change_reason", "")).strip()
        if required_reason:
            records = tuple(getattr(frame, "mass_log", ()))
            matches = [
                record
                for record in records
                if _mass_record_field(record, "reason") == required_reason
            ]
            valid_matches = [record for record in matches if _valid_mass_record(record)]
            details["required_mass_change_reason"] = required_reason
            details["matching_mass_change_records"] = len(matches)
            details["valid_mass_change_records"] = len(valid_matches)
            if not valid_matches:
                failures.append(
                    f"{family_name}: final dataset lacks the reviewed, "
                    "mass-conserving household MassChangeRecord for its SPI "
                    "prior allocation."
                )

        diagnostics[family_name] = details
    return diagnostics, failures


def _mass_record_field(record: object, name: str) -> object:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _valid_mass_record(record: object) -> bool:
    old_total = _mass_record_field(record, "old_total")
    new_total = _mass_record_field(record, "new_total")
    declared_factor = _mass_record_field(record, "declared_factor")
    try:
        old = float(old_total)
        new = float(new_total)
        factor = float(declared_factor)
    except (TypeError, ValueError):
        return False
    return bool(
        _mass_record_field(record, "entity") == "household"
        and np.isfinite(old)
        and old > 0.0
        and np.isfinite(new)
        and np.isclose(old, new, rtol=1e-9, atol=0.0)
        and factor == 1.0
    )


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
    expected_entities = _engine_variable_entities(engine, relevant)
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

    wrong_entities = {
        name: {"actual": entity, "expected": expected_entities[name]}
        for name, (entity, _values) in sorted(present_values.items())
        if entity != expected_entities[name]
    }

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
    for name, mismatch in wrong_entities.items():
        failures.append(
            f"{name}: enhanced-FRS input column is stored on "
            f"{mismatch['actual']!r}, but PolicyEngine-UK owns it on "
            f"{mismatch['expected']!r}; a same-named column on the wrong table "
            "does not satisfy loader coverage."
        )
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
    family_diagnostics, family_failures = _family_effective_mass_diagnostics(
        manifest,
        tables=tables,
        column_values=present_values,
        defaults=defaults,
        effective_weights=effective_weights,
    )
    failures.extend(family_failures)
    family_build_state, family_build_failures = _family_build_state_diagnostics(
        frame,
        manifest,
    )
    failures.extend(family_build_failures)
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
        "wrong_entity_columns": wrong_entities,
        "family_effective_mass": family_diagnostics,
        "family_build_state": family_build_state,
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


def _efrs_input_entities() -> dict[str, str]:
    reference = load_efrs_parity_reference()
    return dict(reference.input_entities)


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
    try:
        candidate_tier = validate_uk_release_tier(
            manifest.candidate_evidence.get("tier")
        )
    except ValueError as exc:
        candidate_tier = None
        failures.append(f"candidate_evidence.tier is invalid: {exc}")
    if candidate_tier is not None and candidate_tier != UK_RELEASE_TIER_FRS:
        failures.append(
            "candidate_evidence.tier must be 'frs' for the bundled "
            f"UKDS-licensed candidate, got {candidate_tier!r}."
        )
    declared = set(manifest.declared_columns)
    surface = set(_efrs_populated_layers())
    reference_entities = _efrs_input_entities()
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

    for name in sorted(RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS):
        if name in manifest.reviewed_exclusions:
            failures.append(
                f"{name}: restored enhanced-FRS reference input cannot return "
                "to a reviewed exclusion."
            )
        elif name not in manifest.required_columns:
            failures.append(
                f"{name}: restored enhanced-FRS reference input must remain required."
            )

    for family_name, family in sorted(manifest.family_coverage.items()):
        source_manifest = str(family["source_manifest"])
        actual_source_sha256 = hashlib.sha256(
            _resource_text(source_manifest).encode("utf-8")
        ).hexdigest()
        if actual_source_sha256 != family["source_manifest_sha256"]:
            failures.append(
                f"{family_name}: {source_manifest} changed without regenerating "
                "release_input_coverage_manifest.json."
            )
        try:
            family_tier = validate_uk_release_tier(family.get("base_candidate_tier"))
        except ValueError as exc:
            family_tier = None
            failures.append(f"{family_name}: base_candidate_tier is invalid: {exc}")
        try:
            source_stage_tier = _source_stage_base_candidate_tier(
                source_manifest,
                stage_name=str(family["stage"]),
            )
        except (OSError, ValueError) as exc:
            source_stage_tier = None
            failures.append(
                f"{family_name}: cannot validate source-stage candidate tier: {exc}"
            )
        if (
            candidate_tier is not None
            and family_tier is not None
            and candidate_tier != family_tier
        ):
            failures.append(
                f"{family_name}: base_candidate_tier {family_tier!r} disagrees "
                f"with candidate_evidence.tier {candidate_tier!r}."
            )
        if (
            family_tier is not None
            and source_stage_tier is not None
            and family_tier != source_stage_tier
        ):
            failures.append(
                f"{family_name}: base_candidate_tier {family_tier!r} disagrees "
                f"with {source_manifest} tier {source_stage_tier!r}."
            )
        requirements = family.get("effective_mass_requirements", {})
        family_status = str(family["status"])
        for column, requirement in sorted(requirements.items()):
            if (
                family_status == _REQUIRED_AT_BUILD_STATUS
                and column not in manifest.required_columns
            ):
                failures.append(
                    f"{family_name}: distributional requirement {column!r} must "
                    "be a required release input while the family is "
                    "required_at_build."
                )
            if (
                family_status == _DEFERRED_UNTIL_RESTORED_STATUS
                and column not in manifest.reviewed_exclusions
            ):
                failures.append(
                    f"{family_name}: deferred distributional requirement "
                    f"{column!r} must remain a reviewed_exclusion until the "
                    "family is restored and promoted."
                )
            requirement_floor = float(requirement["minimum_nondefault_mass_share"])
            if requirement_floor != (
                manifest.effective_mass_coverage.minimum_nondefault_mass_share
            ):
                failures.append(
                    f"{family_name}: {column!r} effective-mass floor disagrees "
                    "with the reviewed UK release policy."
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
                f"loadable variables: {non_inputs}."
            )
        if not non_inputs:
            try:
                live_entities = _engine_variable_entities(engine, declared)
            except ValueError as exc:
                failures.append(str(exc))
            else:
                entity_drift = {
                    name: {
                        "reference": reference_entities[name],
                        "live": live_entities[name],
                    }
                    for name in sorted(surface & declared)
                    if reference_entities[name] != live_entities[name]
                }
                if entity_drift:
                    failures.append(
                        "enhanced-FRS input owning entities disagree with the "
                        f"live PolicyEngine-UK graph: {entity_drift}."
                    )

    if failures:
        raise ValueError(
            "UK release input-column coverage manifest has drifted:\n"
            + "\n".join(f"  - {line}" for line in failures)
        )


def assert_uk_release_input_coverage_build_stages(
    stage_names: Sequence[str],
    *,
    manifest: UKReleaseInputCoverageManifest | None = None,
) -> None:
    """Require every ``required_at_build`` family in the national stage plan."""

    manifest = manifest or load_uk_release_input_coverage_manifest()
    actual = {str(name) for name in stage_names}
    missing = sorted(manifest.required_build_stages - actual)
    if missing:
        raise ValueError(
            "UK national build omits required release family stage(s) "
            f"{missing}; family_coverage status='required_at_build' is an "
            "executable contract, not documentation."
        )
