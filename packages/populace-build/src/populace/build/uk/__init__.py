"""UK build helpers for Populace-owned raw-source and local artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.resources import files

from populace.build.plan import DonorSpec, Stage, StagePlan
from populace.build.source_manifest import (
    SourceManifest,
    SourceStageSpec,
    load_source_manifest,
)
from populace.build.uk.geography_sources import (
    ENGLAND_LAD_REGION_URL,
    ENGLAND_WALES_OA2021_COUNT,
    EW_OA_CONSTITUENCY_URL,
    EW_OA_HIERARCHY_URL,
    EW_OA_LAD23_URL,
    EW_OA_POPULATION_URL,
    MAX_UNMATCHED_ACTIVE_NI_POSTCODE_SHARE,
    NI_DZ2021_COUNT,
    NI_DZ_GEOJSON_ZIP_URL,
    NI_DZ_POPULATION_CSV_URL,
    SCOTLAND_OA2022_COUNT,
    SCOTLAND_OA_CONSTITUENCY_URL,
    SCOTLAND_OA_DZ_IZ_URL,
    SCOTLAND_OA_LAU_ITL_URL,
    SCOTLAND_OA_POPULATION_URL,
    UK_POSTCODE_OA_MAY25_ZIP_URL,
    UK_POSTCODE_PCON_MAY24_ZIP_URL,
    build_complete_uk_geography_crosswalk,
    build_england_wales_crosswalk,
    build_great_britain_crosswalk,
    build_northern_ireland_crosswalk,
    build_official_uk_geography_crosswalk,
    build_scotland_crosswalk,
    infer_ni_dz_constituencies_from_postcodes,
    load_england_lad_region_lookup,
    load_england_wales_oa_constituencies,
    load_england_wales_oa_hierarchy,
    load_england_wales_oa_population,
    load_ew_oa_lad23_lookup,
    load_ni_dz_hierarchy,
    load_ni_dz_population,
    load_scotland_oa_constituencies,
    load_scotland_oa_dz_iz_lookup,
    load_scotland_oa_lau_lookup,
    load_scotland_oa_population,
    load_uk_postcode_constituency_lookup,
    load_uk_postcode_oa_lookup,
    update_england_wales_lad_codes,
    write_geography_crosswalk,
)
from populace.build.uk.local_geography import (
    AREA_TYPE_TO_ROWWISE_HOUSEHOLD_COLUMN,
    LONG_GEOGRAPHY_COLUMNS,
    StackedLocalMatrix,
    align_area_targets,
    area_support_summary,
    assigned_weights_to_long,
    build_assigned_local_matrix,
    build_stacked_local_matrix,
    rowwise_assignment_column,
    sort_households_by_id,
    stacked_design_weights,
    stacked_weights_to_long,
    write_long_geography_weights,
)
from populace.build.uk.local_runner import (
    UKLocalCandidateResult,
    build_local_candidate,
    build_local_candidate_from_dataset,
    build_metric_tables_from_dataset,
    load_metric_tables,
    load_uk_dataset,
    prepare_area_frame,
    prepare_household_frame,
    read_local_table,
    set_simulation_area_group,
    summarize_local_candidate,
    write_local_candidate_outputs,
)
from populace.build.uk.local_solver import (
    StackedLocalSolveResult,
    solve_assigned_local_weights,
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
from populace.build.uk.rowwise_dataset import (
    BENUNIT_ID_COLUMNS,
    HOUSEHOLD_ID_COLUMNS,
    PERSON_ID_COLUMNS,
    UK_SINGLE_YEAR_TABLES,
    UKRowwiseDatasetResult,
    clone_uk_dataset_tables_with_rowwise_geography,
    clone_uk_dataset_with_rowwise_geography,
    validate_uk_rowwise_dataset_tables,
    write_uk_rowwise_dataset,
)
from populace.build.uk.rowwise_geography import (
    AREA_TYPE_TO_CROSSWALK_COLUMN,
    CROSSWALK_COLUMNS,
    FRS_REGION_TO_COUNTRY,
    FRS_REGION_TO_REGION_CODE,
    ROWWISE_GEOGRAPHY_COLUMNS,
    RowwiseGeographyAssignment,
    assign_household_geography,
    clone_entity_frame,
    geography_coverage_summary,
    id_multiplier_for_values,
    prepare_geography_crosswalk,
    validate_geography_coverage,
)
from populace.build.uk.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    DEFAULT_SPI_SUPPORT_HOUSEHOLDS,
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_INCOME_COMPONENT_COLUMNS,
    SPI_INCOME_IMPUTATION_COLUMNS,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    UK_SPI_SUPPORT_STAGE_NAME,
    UKSPISupportResult,
    create_uk_spi_support_tables,
    fill_support_channel_from_source,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.frame import Frame


def _load_uk_source_manifest() -> SourceManifest:
    return load_source_manifest(files(__package__).joinpath("source_stages.json"))


UK_SOURCE_MANIFEST = _load_uk_source_manifest()
UK_STAGE_NAMES: tuple[str, ...] = UK_SOURCE_MANIFEST.plan_stages
_UK_SOURCE_STAGE_MAP = UK_SOURCE_MANIFEST.stage_map()
_UNKNOWN_UK_SOURCE_STAGES = sorted(set(_UK_SOURCE_STAGE_MAP) - set(UK_STAGE_NAMES))
if _UNKNOWN_UK_SOURCE_STAGES:
    raise ValueError(
        "UK source manifest stage(s) are not declared in plan_stages: "
        f"{_UNKNOWN_UK_SOURCE_STAGES}."
    )
UK_SOURCE_STAGE_SPECS: tuple[SourceStageSpec, ...] = tuple(
    _UK_SOURCE_STAGE_MAP[name] for name in UK_STAGE_NAMES if name in _UK_SOURCE_STAGE_MAP
)
UK_DONORS: Mapping[str, DonorSpec] = {
    stage.stage: DonorSpec(
        survey=stage.survey,
        source=stage.source,
        notes=stage.notes,
    )
    for stage in UK_SOURCE_STAGE_SPECS
    if stage.role == "donor"
}
UK_STRUCTURAL_SOURCE_STAGES: tuple[str, ...] = tuple(
    stage.stage for stage in UK_SOURCE_STAGE_SPECS if stage.role != "donor"
)
UK_SOURCE_OUTPUTS: frozenset[str] = frozenset(
    output for stage in UK_SOURCE_STAGE_SPECS for output in stage.outputs
)
UK_SOURCE_OUTPUT_STAGES: Mapping[str, tuple[str, ...]] = {
    output: tuple(
        stage.stage for stage in UK_SOURCE_STAGE_SPECS if output in stage.outputs
    )
    for output in sorted(UK_SOURCE_OUTPUTS)
}
UK_REWRITTEN_SOURCE_OUTPUT_STAGES: Mapping[str, tuple[str, ...]] = {
    output: stages
    for output, stages in UK_SOURCE_OUTPUT_STAGES.items()
    if len(stages) > 1
}
UK_NONNEGATIVE_SOURCE_OUTPUTS: frozenset[str] = frozenset(
    output for stage in UK_SOURCE_STAGE_SPECS for output in stage.nonnegative_outputs
)


def uk_plan(
    implementations: Mapping[str, Callable[[Frame], Frame]],
) -> StagePlan:
    """Assemble the UK build plan from injected stage implementations."""

    missing = [name for name in UK_STAGE_NAMES if name not in implementations]
    if missing:
        raise ValueError(
            f"uk_plan needs an implementation for every declared stage; "
            f"missing {missing}. There are no stubs or fallbacks by design."
        )
    unknown = sorted(set(implementations) - set(UK_STAGE_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown stage implementation(s) {unknown}; declared stages "
            f"are {list(UK_STAGE_NAMES)}."
        )
    return StagePlan(
        Stage(
            name=name,
            transform=implementations[name],
            donor=UK_DONORS.get(name),
        )
        for name in UK_STAGE_NAMES
    )

__all__ = [
    "AGE_BANDS",
    "AREA_TYPES",
    "AREA_TYPE_TO_CROSSWALK_COLUMN",
    "AREA_TYPE_TO_ROWWISE_HOUSEHOLD_COLUMN",
    "BASE_FRS_SUPPORT_CHANNEL",
    "BENUNIT_ID_COLUMNS",
    "COUNTRY_TO_REGION",
    "CROSSWALK_COLUMNS",
    "DEFAULT_SPI_SUPPORT_HOUSEHOLDS",
    "ENGLAND_LAD_REGION_URL",
    "ENGLAND_WALES_OA2021_COUNT",
    "EW_OA_CONSTITUENCY_URL",
    "EW_OA_HIERARCHY_URL",
    "EW_OA_LAD23_URL",
    "EW_OA_POPULATION_URL",
    "FRS_REGION_TO_COUNTRY",
    "FRS_REGION_TO_REGION_CODE",
    "FRS_ONLY_SPI_FILL_PERSON_COLUMNS",
    "FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS",
    "HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN",
    "HOUSEHOLD_ID_COLUMNS",
    "INCOME_VARIABLES",
    "LA_EXTRA_METRICS",
    "LONG_GEOGRAPHY_COLUMNS",
    "MAX_UNMATCHED_ACTIVE_NI_POSTCODE_SHARE",
    "NI_DZ2021_COUNT",
    "NI_DZ_GEOJSON_ZIP_URL",
    "NI_DZ_POPULATION_CSV_URL",
    "PERSON_ID_COLUMNS",
    "ROWWISE_GEOGRAPHY_COLUMNS",
    "RowwiseGeographyAssignment",
    "SCOTLAND_OA2022_COUNT",
    "SCOTLAND_OA_CONSTITUENCY_URL",
    "SCOTLAND_OA_DZ_IZ_URL",
    "SCOTLAND_OA_LAU_ITL_URL",
    "SCOTLAND_OA_POPULATION_URL",
    "SPI_INCOME_COMPONENT_COLUMNS",
    "SPI_INCOME_IMPUTATION_COLUMNS",
    "SPI_SYNTHETIC_SUPPORT_CHANNEL",
    "StackedLocalMatrix",
    "StackedLocalSolveResult",
    "UK_POSTCODE_OA_MAY25_ZIP_URL",
    "UK_POSTCODE_PCON_MAY24_ZIP_URL",
    "UK_DONORS",
    "UKLocalCandidateResult",
    "UK_NONNEGATIVE_SOURCE_OUTPUTS",
    "UKRowwiseDatasetResult",
    "UKSPISupportResult",
    "UK_SOURCE_MANIFEST",
    "UK_SOURCE_OUTPUTS",
    "UK_SOURCE_OUTPUT_STAGES",
    "UK_SOURCE_STAGE_SPECS",
    "UK_REWRITTEN_SOURCE_OUTPUT_STAGES",
    "UK_SINGLE_YEAR_TABLES",
    "UK_SPI_SUPPORT_STAGE_NAME",
    "UK_STAGE_NAMES",
    "UK_STRUCTURAL_SOURCE_STAGES",
    "align_area_targets",
    "assigned_weights_to_long",
    "area_support_summary",
    "area_groups_from_codes",
    "assign_household_geography",
    "build_local_candidate",
    "build_local_candidate_from_dataset",
    "build_assigned_local_matrix",
    "build_complete_uk_geography_crosswalk",
    "build_england_wales_crosswalk",
    "build_great_britain_crosswalk",
    "build_metric_tables_from_dataset",
    "build_northern_ireland_crosswalk",
    "build_official_uk_geography_crosswalk",
    "build_scotland_crosswalk",
    "build_stacked_local_matrix",
    "clone_entity_frame",
    "clone_uk_dataset_tables_with_rowwise_geography",
    "clone_uk_dataset_with_rowwise_geography",
    "compute_household_metrics",
    "create_uk_spi_support_tables",
    "fill_support_channel_from_source",
    "geography_coverage_summary",
    "id_multiplier_for_values",
    "infer_ni_dz_constituencies_from_postcodes",
    "load_england_lad_region_lookup",
    "load_england_wales_oa_constituencies",
    "load_england_wales_oa_hierarchy",
    "load_england_wales_oa_population",
    "load_ew_oa_lad23_lookup",
    "load_metric_tables",
    "load_ni_dz_hierarchy",
    "load_ni_dz_population",
    "load_scotland_oa_constituencies",
    "load_scotland_oa_dz_iz_lookup",
    "load_scotland_oa_lau_lookup",
    "load_scotland_oa_population",
    "load_uk_postcode_constituency_lookup",
    "load_uk_postcode_oa_lookup",
    "load_uk_dataset",
    "metric_names",
    "metric_tables_by_area_group",
    "prepare_area_frame",
    "prepare_geography_crosswalk",
    "prepare_household_frame",
    "read_local_table",
    "rowwise_assignment_column",
    "set_simulation_area_group",
    "solve_assigned_local_weights",
    "solve_stacked_local_weights",
    "sort_households_by_id",
    "stacked_design_weights",
    "stacked_weights_to_long",
    "summarize_local_candidate",
    "support_channel_column",
    "support_clone_index_column",
    "support_source_id_column",
    "uk_plan",
    "update_england_wales_lad_codes",
    "validate_uk_rowwise_dataset_tables",
    "validate_geography_coverage",
    "write_geography_crosswalk",
    "write_local_candidate_outputs",
    "write_long_geography_weights",
    "write_uk_rowwise_dataset",
]
