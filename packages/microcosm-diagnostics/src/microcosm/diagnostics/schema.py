"""Typed calibration-diagnostics schemas shared by producers and consumers."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 8
SUPPORTED_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS = frozenset({6, 7, 8})


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("text must contain a non-whitespace character")
    return value


NonBlankText = Annotated[str, AfterValidator(_require_non_blank)]


class DiagnosticsModel(BaseModel):
    """Base for immutable schema records with no undeclared fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Selector(DiagnosticsModel):
    kind: Literal["column", "callable"]
    name: NonBlankText


class HierarchyNode(DiagnosticsModel):
    id: NonBlankText
    label: NonBlankText


class HierarchyCategory(HierarchyNode):
    provider_id: NonBlankText


class HierarchyGeography(HierarchyNode):
    level: NonBlankText


class HierarchyDimension(DiagnosticsModel):
    id: NonBlankText
    label: NonBlankText
    value_id: NonBlankText
    value_label: NonBlankText


class CalibrationHierarchy(DiagnosticsModel):
    provider: HierarchyNode
    category: HierarchyCategory
    geography: HierarchyGeography
    dimensions: list[HierarchyDimension]
    target: HierarchyNode

    @model_validator(mode="after")
    def category_matches_provider(self) -> CalibrationHierarchy:
        if self.category.provider_id != self.provider.id:
            raise ValueError("hierarchy category provider_id must match provider id")
        dimension_ids = [dimension.id for dimension in self.dimensions]
        if len(dimension_ids) != len(set(dimension_ids)):
            raise ValueError("hierarchy dimension ids must be unique")
        return self


class RegistryTargetMetadata(DiagnosticsModel):
    family: str
    se: float | None
    signed: bool
    notes: str


class TargetDiagnosticV8(DiagnosticsModel):
    name: NonBlankText
    target_name: NonBlankText
    period: int | NonBlankText
    entity: NonBlankText
    measure: Selector
    filter: Selector | None
    source: NonBlankText
    metadata: dict[str, JsonValue]
    target: float | None
    compiled_target: float | None
    initial_estimate: float | None
    final_estimate: float | None
    relative_error: float | None
    within_tolerance: bool | None
    hierarchy: CalibrationHierarchy
    registry: RegistryTargetMetadata | None = None
    target_loss_weight: float | None = None
    target_loss_weight_share: float | None = None
    target_loss_scale: float | None = None
    final_capped_scaled_error: float | None = None
    final_loss_contribution: float | None = None

    @model_validator(mode="after")
    def hierarchy_target_matches_row(self) -> TargetDiagnosticV8:
        if self.hierarchy.target.id != self.target_name:
            raise ValueError("hierarchy target id must match target_name")
        return self


class TargetSurfaceMatrix(DiagnosticsModel):
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    nnz: int = Field(ge=0)


class TargetSurface(DiagnosticsModel):
    schema_version: Literal[1]
    weight_entity: NonBlankText
    n_targets: int = Field(gt=0)
    n_records: int = Field(gt=0)
    constraint_matrix: TargetSurfaceMatrix
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    names_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    values_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def reconcile_matrix_shape(self) -> TargetSurface:
        if self.constraint_matrix.rows != self.n_targets:
            raise ValueError("constraint_matrix.rows must equal n_targets")
        if self.constraint_matrix.columns != self.n_records:
            raise ValueError("constraint_matrix.columns must equal n_records")
        if self.constraint_matrix.nnz > (
            self.constraint_matrix.rows * self.constraint_matrix.columns
        ):
            raise ValueError("constraint_matrix.nnz exceeds the matrix capacity")
        return self


class TargetRegistryRef(DiagnosticsModel):
    country: NonBlankText
    version: NonBlankText
    n_specs: int = Field(gt=0)


class SkippedTarget(DiagnosticsModel):
    name: NonBlankText
    reason: NonBlankText


class DiagnosticsWarning(DiagnosticsModel):
    code: NonBlankText
    severity: Literal["warning"]
    message: NonBlankText


class PushedOutTarget(DiagnosticsModel):
    name: NonBlankText
    init_rel: float
    final_rel: float


class PastCapCensus(DiagnosticsModel):
    cap: float = Field(gt=0)
    scale_basis: NonBlankText
    n_targets: int = Field(ge=0)
    initial_past_cap: int = Field(ge=0)
    final_past_cap: int = Field(ge=0)
    escaped: int = Field(ge=0)
    frozen: int = Field(ge=0)
    pushed_out: int = Field(ge=0)
    pushed_out_rows: list[PushedOutTarget]


class TargetLossBasis(DiagnosticsModel):
    formula: NonBlankText
    cap: float = Field(gt=0)
    target_count: int = Field(ge=0)
    total_target_weight: float = Field(gt=0)
    weight_kind: NonBlankText
    scale_kind: NonBlankText
    hash_algorithm: NonBlankText
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationDiagnosticsV8(DiagnosticsModel):
    """The only calibration-diagnostics schema emitted by current builds."""

    schema_version: Literal[8]
    weight_entity: NonBlankText
    options: dict[str, JsonValue]
    target_surface: TargetSurface
    target_registry: TargetRegistryRef
    l0_lambda: float | None
    n_nonzero: int = Field(ge=0)
    n_records: int = Field(gt=0)
    initial_loss: float | None
    final_loss: float | None
    fraction_within_10pct: float | None
    effective_sample_size: float | None
    realized_max_weight_ratio: float | None
    top_1pct_weight_share: float | None
    loss_trajectory: list[float | None]
    skipped: list[SkippedTarget]
    past_cap_census: PastCapCensus | None
    diagnostic_warnings: list[DiagnosticsWarning]
    targets: list[TargetDiagnosticV8]
    target_loss_basis: TargetLossBasis | None = None
    build: dict[str, JsonValue] | None = None
    uk_diagnostics: dict[str, JsonValue] | None = None

    @field_validator(
        "l0_lambda",
        "initial_loss",
        "final_loss",
        "fraction_within_10pct",
        "effective_sample_size",
        "realized_max_weight_ratio",
        "top_1pct_weight_share",
    )
    @classmethod
    def finite_optional_scalar(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("diagnostic scalars must be finite or null")
        return value

    @model_validator(mode="after")
    def reconcile_counts(self) -> CalibrationDiagnosticsV8:
        if self.target_surface.n_targets != len(self.targets):
            raise ValueError("target_surface.n_targets must equal len(targets)")
        if self.target_registry.n_specs < len(self.targets):
            raise ValueError("target_registry.n_specs cannot be less than len(targets)")
        if self.target_surface.n_records != self.n_records:
            raise ValueError("target_surface.n_records must equal n_records")
        if self.n_nonzero > self.n_records:
            raise ValueError("n_nonzero cannot exceed n_records")
        return self


class LegacyCalibrationDiagnostics(DiagnosticsModel):
    """Typed common envelope retained for immutable schema-6/7 releases."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=False)

    schema_version: Literal[6, 7]
    weight_entity: str = Field(min_length=1)
    options: dict[str, JsonValue]
    target_surface: dict[str, JsonValue]
    target_registry: dict[str, JsonValue]
    targets: list[dict[str, JsonValue]]
    loss_trajectory: list[float | None]
    skipped: list[dict[str, JsonValue]]


CalibrationDiagnostics = Annotated[
    CalibrationDiagnosticsV8 | LegacyCalibrationDiagnostics,
    Field(discriminator="schema_version"),
]
CALIBRATION_DIAGNOSTICS_ADAPTER = TypeAdapter(CalibrationDiagnostics)


def parse_calibration_diagnostics(value: object) -> CalibrationDiagnostics:
    """Validate a decoded diagnostics document and return its typed model."""

    return CALIBRATION_DIAGNOSTICS_ADAPTER.validate_python(value)


def calibration_diagnostics_json_schema() -> dict[str, JsonValue]:
    """Return the machine-readable schema generated from the typed models."""

    return CALIBRATION_DIAGNOSTICS_ADAPTER.json_schema()
