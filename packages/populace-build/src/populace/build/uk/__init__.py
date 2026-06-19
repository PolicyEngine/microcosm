"""UK build helpers for Populace-owned local-geography artifacts."""

from populace.build.uk.local_geography import (
    LONG_GEOGRAPHY_COLUMNS,
    StackedLocalMatrix,
    align_area_targets,
    area_support_summary,
    build_stacked_local_matrix,
    sort_households_by_id,
    stacked_design_weights,
    stacked_weights_to_long,
    write_long_geography_weights,
)
from populace.build.uk.local_solver import (
    StackedLocalSolveResult,
    solve_stacked_local_weights,
)
from populace.build.uk.local_targets import (
    AGE_BANDS,
    AREA_TYPES,
    COUNTRY_TO_REGION,
    INCOME_VARIABLES,
    LA_EXTRA_METRICS,
    area_groups_from_codes,
    compute_household_metrics,
    metric_names,
    metric_tables_by_area_group,
)

__all__ = [
    "AGE_BANDS",
    "AREA_TYPES",
    "COUNTRY_TO_REGION",
    "INCOME_VARIABLES",
    "LA_EXTRA_METRICS",
    "LONG_GEOGRAPHY_COLUMNS",
    "StackedLocalMatrix",
    "StackedLocalSolveResult",
    "align_area_targets",
    "area_support_summary",
    "area_groups_from_codes",
    "build_stacked_local_matrix",
    "compute_household_metrics",
    "metric_names",
    "metric_tables_by_area_group",
    "solve_stacked_local_weights",
    "sort_households_by_id",
    "stacked_design_weights",
    "stacked_weights_to_long",
    "write_long_geography_weights",
]
