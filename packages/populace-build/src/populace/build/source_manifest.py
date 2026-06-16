"""Generic source-stage manifests for country build plans.

Country packages own source content as data: JSON manifests describing source
artifacts, required transformations, outputs, and validation requirements. The
Python here is the shared interpreter contract only; it is intentionally not a
country-specific donor loader.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_SOURCE_OPERATION_KINDS",
    "FORBIDDEN_EXECUTABLE_LOADER_KEYS",
    "FORBIDDEN_EXECUTABLE_OPERATION_KINDS",
    "FORBIDDEN_SOURCE_DEPENDENCIES",
    "SourceManifest",
    "SourceOperationSpec",
    "SourceStageSpec",
    "load_source_manifest",
]


FORBIDDEN_SOURCE_DEPENDENCIES = (
    "policyengine_" + "us_data",
    "policyengine-" + "us-data",
    "policyengine_" + "uk_data",
    "policyengine-" + "uk-data",
)

ALLOWED_SOURCE_OPERATION_KINDS = frozenset(
    {
        "aggregate_person_to_household",
        "assign_by_plan_type",
        "convert_interest_to_structural_mortgage_inputs",
        "derive",
        "derive_mortgage_balance_hints",
        "disaggregate_aggregate_records",
        "fit_labor_market_models",
        "fit_tip_income_model",
        "fit_vehicle_model",
        "fit_weighted_imputer",
        "fit_weighted_qrf",
        "fold_into",
        "head_carry",
        "join",
        "read_table",
        "read_tables",
        "replace_sentinels",
        "support_clip",
        "uprate",
        "zero_when_false",
    }
)

FORBIDDEN_EXECUTABLE_OPERATION_KINDS = frozenset(
    {
        "callable",
        "exec",
        "function",
        "import",
        "import_module",
        "module",
        "python",
        "python_callable",
        "python_function",
        "python_module",
    }
)

FORBIDDEN_EXECUTABLE_LOADER_KEYS = frozenset(
    {
        "callable",
        "entrypoint",
        "function",
        "import",
        "loader",
        "module",
        "python",
    }
)


@dataclass(frozen=True)
class SourceOperationSpec:
    """One declarative source operation.

    The operation names are generic primitives such as ``read_table``,
    ``replace_sentinels``, ``derive``, or ``fit_weighted_qrf``. Source manifests
    must not point at country-specific Python modules or incumbent data-package
    helpers.
    """

    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceOperationSpec:
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("source operation requires a non-empty string 'kind'.")
        parameters = {k: v for k, v in raw.items() if k != "kind"}
        _reject_executable_loader_shape(kind, parameters)
        _reject_incumbent_dependencies(parameters, context=f"operation {kind!r}")
        return cls(kind=kind, parameters=parameters)


@dataclass(frozen=True)
class SourceStageSpec:
    """Declarative source-stage contract for one build stage."""

    stage: str
    survey: str
    source: str
    grain: str
    artifacts: tuple[Mapping[str, Any], ...]
    operations: tuple[SourceOperationSpec, ...]
    outputs: tuple[str, ...]
    nonnegative_outputs: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceStageSpec:
        required = ("stage", "survey", "source", "grain", "operations", "outputs")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"source stage is missing required key(s): {missing}.")
        for key in ("stage", "survey", "source", "grain"):
            if not isinstance(raw[key], str) or not raw[key]:
                raise ValueError(
                    f"source stage key {key!r} must be a non-empty string."
                )
        artifacts = tuple(_require_mapping_sequence(raw.get("artifacts", ())))
        operations = tuple(
            SourceOperationSpec.from_mapping(operation)
            for operation in _require_mapping_sequence(raw["operations"])
        )
        outputs = tuple(_require_string_sequence(raw["outputs"], key="outputs"))
        nonnegative_outputs = tuple(
            _require_string_sequence(
                raw.get("nonnegative_outputs", ()),
                key="nonnegative_outputs",
            )
        )
        unknown_nonnegative = sorted(set(nonnegative_outputs) - set(outputs))
        if unknown_nonnegative:
            raise ValueError(
                f"stage {raw['stage']!r} marks nonnegative output(s) not in outputs: "
                f"{unknown_nonnegative}."
            )
        notes = raw.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("source stage 'notes' must be a string when provided.")
        _reject_executable_parameter_keys(raw, context=f"stage {raw['stage']!r}")
        _reject_incumbent_dependencies(raw, context=f"stage {raw['stage']!r}")
        return cls(
            stage=raw["stage"],
            survey=raw["survey"],
            source=raw["source"],
            grain=raw["grain"],
            artifacts=artifacts,
            operations=operations,
            outputs=outputs,
            nonnegative_outputs=nonnegative_outputs,
            notes=notes,
        )


@dataclass(frozen=True)
class SourceManifest:
    """A country source manifest loaded from packaged JSON."""

    country: str
    version: int
    policy: str
    stages: tuple[SourceStageSpec, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceManifest:
        country = raw.get("country")
        version = raw.get("version")
        policy = raw.get("policy", "")
        if not isinstance(country, str) or not country:
            raise ValueError("source manifest requires a non-empty 'country'.")
        if not isinstance(version, int) or version < 1:
            raise ValueError("source manifest requires positive integer 'version'.")
        if not isinstance(policy, str) or not policy:
            raise ValueError("source manifest requires a non-empty 'policy'.")
        stages = tuple(
            SourceStageSpec.from_mapping(stage)
            for stage in _require_mapping_sequence(raw.get("stages", ()))
        )
        names = [stage.stage for stage in stages]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate source stage spec(s): {duplicates}.")
        _reject_executable_parameter_keys(raw, context=f"{country} source manifest")
        _reject_incumbent_dependencies(raw, context=f"{country} source manifest")
        return cls(country=country, version=version, policy=policy, stages=stages)

    def stage_map(self) -> Mapping[str, SourceStageSpec]:
        return {stage.stage: stage for stage in self.stages}


def load_source_manifest(resource: Any) -> SourceManifest:
    """Load and validate a source manifest from a path-like resource."""
    if hasattr(resource, "read_text"):
        text = resource.read_text(encoding="utf-8")
    else:
        text = Path(resource).read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, Mapping):
        raise ValueError("source manifest root must be a JSON object.")
    return SourceManifest.from_mapping(raw)


def _require_mapping_sequence(raw: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("expected a list of objects.")
    result = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("expected every list item to be an object.")
        result.append(dict(item))
    return tuple(result)


def _require_string_sequence(raw: object, *, key: str) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError(f"{key} must be a list of strings.")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{key} must contain only non-empty strings.")
        values.append(item)
    return tuple(values)


def _reject_incumbent_dependencies(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_incumbent_dependencies(key, context=context)
            _reject_incumbent_dependencies(nested, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_incumbent_dependencies(nested, context=context)
        return
    if not isinstance(value, str):
        return
    lowered = value.lower()
    for dependency in FORBIDDEN_SOURCE_DEPENDENCIES:
        if dependency in lowered:
            raise ValueError(
                f"{context} references forbidden incumbent dependency {dependency!r}."
            )


def _reject_executable_loader_shape(kind: str, parameters: Mapping[str, Any]) -> None:
    normalized_kind = _normalize_manifest_key(kind)
    if normalized_kind in FORBIDDEN_EXECUTABLE_OPERATION_KINDS:
        raise ValueError(
            f"source operation {kind!r} is executable-loader content, not a "
            "declarative source operation."
        )
    if normalized_kind not in ALLOWED_SOURCE_OPERATION_KINDS:
        raise ValueError(
            f"source operation {kind!r} is not in the allowed manifest operation "
            "vocabulary."
        )
    _reject_executable_parameter_keys(parameters, context=f"operation {kind!r}")


def _reject_executable_parameter_keys(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                normalized = _normalize_manifest_key(key)
                if normalized in FORBIDDEN_EXECUTABLE_LOADER_KEYS:
                    raise ValueError(
                        f"{context} uses executable-loader key {key!r}; source "
                        "manifests must be declarative."
                    )
            _reject_executable_parameter_keys(nested, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_executable_parameter_keys(nested, context=context)


def _normalize_manifest_key(value: str) -> str:
    return value.lower().replace("-", "_").strip()
