"""Typed calibration-diagnostics schemas shared by producers and consumers."""

from __future__ import annotations

import math
from typing import Annotated, Final, Literal

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

CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION: Final[Literal[8]] = 8
SUPPORTED_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS = frozenset({6, 7, 8})
UK_DIAGNOSTICS_SCHEMA_VERSION: Final[Literal[1]] = 1


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("text must contain a non-whitespace character")
    return value


NonBlankText = Annotated[str, AfterValidator(_require_non_blank)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]
PositiveFiniteFloat = Annotated[
    float,
    Field(gt=0, allow_inf_nan=False),
]
Fraction = Annotated[
    float,
    Field(ge=0, le=1, allow_inf_nan=False),
]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
JsonScalar = str | bool | int | FiniteFloat | None


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


UKGeographyLevel = Literal[
    "national",
    "region",
    "country",
    "local_authority",
    "constituency",
]
UKLocalGeographyLevel = Literal["local_authority", "constituency"]
UKCountry = Literal["England", "Northern Ireland", "Scotland", "Wales"]
UK_TARGET_GEOGRAPHY_LEVELS: Final[tuple[UKGeographyLevel, ...]] = (
    "national",
    "region",
    "country",
    "local_authority",
    "constituency",
)


class UKWeightSummary(DiagnosticsModel):
    n_records: PositiveInt
    positive_weight_records: NonNegativeInt
    zero_weight_records: NonNegativeInt
    total_weight: NonNegativeFiniteFloat
    effective_sample_size: NonNegativeFiniteFloat
    ess_fraction: Fraction
    median_positive_weight: PositiveFiniteFloat | None
    max_weight: NonNegativeFiniteFloat
    max_to_median_positive_weight: (
        Annotated[
            float,
            Field(ge=1, allow_inf_nan=False),
        ]
        | None
    )
    top_1pct_weight_share: Fraction

    @model_validator(mode="after")
    def reconcile_weight_summary(self) -> UKWeightSummary:
        if self.positive_weight_records + self.zero_weight_records != self.n_records:
            raise ValueError(
                "UK positive- and zero-weight record counts must sum to n_records"
            )
        if self.effective_sample_size > self.n_records:
            raise ValueError("UK effective_sample_size cannot exceed n_records")
        if not math.isclose(
            self.ess_fraction,
            self.effective_sample_size / self.n_records,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "UK ess_fraction must equal effective_sample_size/n_records"
            )
        if self.positive_weight_records == 0:
            if (
                self.median_positive_weight is not None
                or self.max_to_median_positive_weight is not None
            ):
                raise ValueError(
                    "UK all-zero weights require null median and max-to-median ratio"
                )
        elif (
            self.median_positive_weight is None
            or self.max_to_median_positive_weight is None
        ):
            raise ValueError(
                "UK positive weights require median and max-to-median ratio"
            )
        elif not math.isclose(
            self.max_to_median_positive_weight,
            self.max_weight / self.median_positive_weight,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "UK max-to-median ratio must equal max_weight/median_positive_weight"
            )
        return self


class UKZeroWeightStratum(DiagnosticsModel):
    stratum: dict[NonBlankText, JsonScalar] = Field(min_length=1)
    rows: NonNegativeInt
    positive_weight_rows: NonNegativeInt
    zero_weight_rows: NonNegativeInt
    weight_sum: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def reconcile_row_counts(self) -> UKZeroWeightStratum:
        if self.positive_weight_rows + self.zero_weight_rows != self.rows:
            raise ValueError("UK zero-weight stratum row counts do not reconcile")
        return self


class UKGeographyPassRate(DiagnosticsModel):
    geography_level: UKGeographyLevel
    n_targets: NonNegativeInt
    n_scored: NonNegativeInt
    n_skipped: NonNegativeInt
    n_within_10pct: NonNegativeInt
    pass_rate: Fraction | None

    @model_validator(mode="after")
    def reconcile_pass_rate(self) -> UKGeographyPassRate:
        if self.n_scored + self.n_skipped != self.n_targets:
            raise ValueError("UK geography scored and skipped counts do not reconcile")
        if self.n_within_10pct > self.n_scored:
            raise ValueError(
                "UK geography passing targets cannot exceed scored targets"
            )
        expected = self.n_within_10pct / self.n_targets if self.n_targets else None
        if expected is None:
            if self.pass_rate is not None:
                raise ValueError("UK empty geography rows require a null pass_rate")
        elif self.pass_rate is None or not math.isclose(
            self.pass_rate,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "UK geography pass_rate must equal n_within_10pct/n_targets"
            )
        return self


class UKWeakestFamily(DiagnosticsModel):
    family: NonBlankText
    n_targets: PositiveInt
    n_within_10pct: NonNegativeInt
    pass_rate: Fraction
    worst_target: NonBlankText
    worst_abs_relative_error: NonNegativeFiniteFloat
    loss_contribution: NonNegativeFiniteFloat
    loss_share: Fraction

    @model_validator(mode="after")
    def reconcile_pass_rate(self) -> UKWeakestFamily:
        if self.n_within_10pct > self.n_targets:
            raise ValueError("UK family passing targets cannot exceed total targets")
        if not math.isclose(
            self.pass_rate,
            self.n_within_10pct / self.n_targets,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("UK family pass_rate must equal n_within_10pct/n_targets")
        return self


class UKAreaFit(DiagnosticsModel):
    geography_level: UKLocalGeographyLevel
    area_code: NonBlankText
    country: UKCountry
    n_targets: PositiveInt
    n_within_10pct: NonNegativeInt
    pass_rate: Fraction
    worst_target: NonBlankText
    worst_abs_relative_error: NonNegativeFiniteFloat
    loss_contribution: NonNegativeFiniteFloat
    nonzero_households: NonNegativeInt
    nonzero_source_households: NonNegativeInt
    effective_sample_size: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def reconcile_pass_rate(self) -> UKAreaFit:
        if self.n_within_10pct > self.n_targets:
            raise ValueError("UK area passing targets cannot exceed total targets")
        if not math.isclose(
            self.pass_rate,
            self.n_within_10pct / self.n_targets,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("UK area pass_rate must equal n_within_10pct/n_targets")
        return self


class UKCountryFit(DiagnosticsModel):
    country: UKCountry
    geography_level: UKLocalGeographyLevel
    n_areas: PositiveInt
    n_targets: PositiveInt
    n_within_10pct: NonNegativeInt
    pass_rate: Fraction
    worst_target: NonBlankText
    worst_abs_relative_error: NonNegativeFiniteFloat
    loss_contribution: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def reconcile_pass_rate(self) -> UKCountryFit:
        if self.n_within_10pct > self.n_targets:
            raise ValueError("UK country passing targets cannot exceed total targets")
        if not math.isclose(
            self.pass_rate,
            self.n_within_10pct / self.n_targets,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("UK country pass_rate must equal n_within_10pct/n_targets")
        return self


class UKWeakestAreasByFit(DiagnosticsModel):
    limit: PositiveInt
    n_areas_scored: NonNegativeInt
    bottom_by_fit: list[UKAreaFit]
    countries: list[UKCountryFit]

    @model_validator(mode="after")
    def reconcile_area_counts(self) -> UKWeakestAreasByFit:
        if len(self.bottom_by_fit) > self.limit:
            raise ValueError("UK bottom-by-fit rows cannot exceed limit")
        if len(self.bottom_by_fit) > self.n_areas_scored:
            raise ValueError("UK bottom-by-fit rows cannot exceed n_areas_scored")
        return self


class UKRotatedHoldoutFold(DiagnosticsModel):
    fold: NonNegativeInt
    n_train_targets: NonNegativeInt
    n_holdout_targets: PositiveInt
    holdout_target_indices: list[NonNegativeInt]
    training_national_rows: NonNegativeInt
    holdout_loss: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def reconcile_holdout_indices(self) -> UKRotatedHoldoutFold:
        if len(self.holdout_target_indices) != self.n_holdout_targets:
            raise ValueError(
                "UK holdout target-index count must equal n_holdout_targets"
            )
        if len(set(self.holdout_target_indices)) != len(self.holdout_target_indices):
            raise ValueError("UK holdout target indices must be unique")
        return self


class UKMeasuredRotatedHoldout(DiagnosticsModel):
    report_only: Literal[True]
    method: Literal["rotated_folds"]
    target_loss_cap: PositiveFiniteFloat
    loss_weight_scale: Literal["held_local_grains_only"]
    target_weight_rule: Literal["uniform", "grain_equal"]
    population: Literal["held_out_local_targets"]
    grains: list[Literal["constituency", "local_authority", "la"]] = Field(min_length=1)
    n_folds: Annotated[int, Field(ge=2)]
    seed: int
    solve_seed: int
    mean_holdout_loss: NonNegativeFiniteFloat
    worst_holdout_loss: NonNegativeFiniteFloat
    fold_losses: list[NonNegativeFiniteFloat]
    folds: list[UKRotatedHoldoutFold]

    @model_validator(mode="after")
    def reconcile_folds(self) -> UKMeasuredRotatedHoldout:
        if len(set(self.grains)) != len(self.grains):
            raise ValueError("UK holdout grains must be unique")
        if len(self.fold_losses) != self.n_folds or len(self.folds) != self.n_folds:
            raise ValueError("UK holdout must carry one loss and row per fold")
        if {fold.fold for fold in self.folds} != set(range(self.n_folds)):
            raise ValueError("UK holdout fold identifiers must cover 0..n_folds-1")
        if any(
            not math.isclose(
                fold.holdout_loss,
                self.fold_losses[fold.fold],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for fold in self.folds
        ):
            raise ValueError("UK holdout fold rows must match fold_losses")
        if not math.isclose(
            self.mean_holdout_loss,
            math.fsum(self.fold_losses) / self.n_folds,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("UK mean holdout loss must close over fold_losses")
        if not math.isclose(
            self.worst_holdout_loss,
            max(self.fold_losses),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("UK worst holdout loss must equal the worst fold loss")
        return self


class UKSkippedRotatedHoldout(DiagnosticsModel):
    skipped: Literal[True]


UKRotatedHoldout = UKMeasuredRotatedHoldout | UKSkippedRotatedHoldout


class UKDiagnosticsV1(DiagnosticsModel):
    schema_version: Literal[UK_DIAGNOSTICS_SCHEMA_VERSION]
    weights: UKWeightSummary
    zero_weight_rows_by_stratum: list[UKZeroWeightStratum] = Field(min_length=1)
    target_pass_rates_by_geography_level: list[UKGeographyPassRate]
    target_observation_basis: dict[NonBlankText, NonBlankText]
    weakest_families: list[UKWeakestFamily] | None = None
    weakest_areas_by_fit: UKWeakestAreasByFit | None = None
    rotated_holdout: UKRotatedHoldout | None = None

    @model_validator(mode="after")
    def reconcile_uk_diagnostics(self) -> UKDiagnosticsV1:
        stratum_rows = sum(row.rows for row in self.zero_weight_rows_by_stratum)
        stratum_positive = sum(
            row.positive_weight_rows for row in self.zero_weight_rows_by_stratum
        )
        stratum_zero = sum(
            row.zero_weight_rows for row in self.zero_weight_rows_by_stratum
        )
        if stratum_rows != self.weights.n_records:
            raise ValueError(
                "UK zero-weight stratum rows do not reconcile to weights.n_records"
            )
        if stratum_positive != self.weights.positive_weight_records:
            raise ValueError(
                "UK positive stratum rows do not reconcile to positive_weight_records"
            )
        if stratum_zero != self.weights.zero_weight_records:
            raise ValueError(
                "UK zero-weight stratum rows do not reconcile to zero_weight_records"
            )
        levels = [
            row.geography_level for row in self.target_pass_rates_by_geography_level
        ]
        if len(levels) != len(set(levels)):
            raise ValueError("UK geography pass-rate levels must be unique")
        expected_levels = set(UK_TARGET_GEOGRAPHY_LEVELS)
        missing = sorted(expected_levels - set(levels))
        unexpected = sorted(set(levels) - expected_levels)
        if missing or unexpected:
            raise ValueError(
                "UK geography pass rates are missing level(s) or contain "
                f"unexpected levels: missing={missing}, unexpected={unexpected}"
            )
        if (self.weakest_families is None) != (self.weakest_areas_by_fit is None):
            raise ValueError(
                "UK local fit diagnostics require both family and area summaries"
            )
        return self


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
    uk_diagnostics: UKDiagnosticsV1 | None = None

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
        if self.target_registry.country == "uk":
            if self.uk_diagnostics is None:
                raise ValueError(
                    "UK calibration diagnostics require a uk_diagnostics block"
                )
            uk = self.uk_diagnostics
            if uk.weights.n_records != self.n_records:
                raise ValueError("UK weights.n_records must match top-level n_records")
            if self.effective_sample_size is None or not math.isclose(
                uk.weights.effective_sample_size,
                self.effective_sample_size,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "UK weights.effective_sample_size must match the top-level value"
                )
            if self.top_1pct_weight_share is None or not math.isclose(
                uk.weights.top_1pct_weight_share,
                self.top_1pct_weight_share,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "UK weights.top_1pct_weight_share must match the top-level value"
                )
            total_targets = sum(
                row.n_targets for row in uk.target_pass_rates_by_geography_level
            )
            total_scored = sum(
                row.n_scored for row in uk.target_pass_rates_by_geography_level
            )
            total_skipped = sum(
                row.n_skipped for row in uk.target_pass_rates_by_geography_level
            )
            if total_targets != self.target_registry.n_specs:
                raise ValueError(
                    "UK geography target counts must match target_registry.n_specs"
                )
            if total_scored != len(self.targets):
                raise ValueError(
                    "UK geography scored counts must match the target row count"
                )
            if total_skipped != len(self.skipped):
                raise ValueError(
                    "UK geography skipped counts must match the skipped row count"
                )
        elif self.uk_diagnostics is not None:
            raise ValueError(
                "uk_diagnostics is only valid when target_registry.country is 'uk'"
            )
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
