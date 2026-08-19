import json
from dataclasses import asdict

import pytest

from microcosm.build.gates import (
    TargetCoverageRequirement,
    target_profile_coverage_gate,
)
from microcosm.build.ledger_targets import (
    LedgerTargetMapping,
    LedgerTargetReference,
    apply_ledger_target_profile,
    compile_ledger_target_references,
    ledger_target_registry_parity_report,
    select_ledger_targets,
    select_ledger_targets_from_jsonl,
)
from microcosm.calibrate import TargetRegistry, TargetSpec


def _ledger_fact(**overrides):
    fact = {
        "fact_key": "ledger.fact.v1:abc123",
        "source_record_id": "irs_soi.ty2023.table_1_1.all.adjusted_gross_income",
        "value": 15_286_017_359_000,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {
            "level": "country",
            "id": "0100000US",
            "name": "United States",
            "vintage": "2020_census",
        },
        "entity": {"name": "tax_unit", "role": "filing_unit"},
        "measure": {
            "concept": "us:statutes/26/62#adjusted_gross_income",
            "unit": "usd",
            "source_concept": "irs_soi.adjusted_gross_income",
            "concept_relation": "exact",
            "concept_authority": "ledger-us",
            "legal_vintage": "tax_year_2023",
        },
        "aggregation": {"method": "sum"},
        "source": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_file": "23in11si.xls",
            "url": "https://www.irs.gov/pub/irs-soi/23in11si.xls",
            "vintage": "tax_year_2023",
        },
        "filters": {"income_range": "all", "filing_status": "all"},
        "domain": "all_individual_income_tax_returns",
        "layout": {
            "record_set_id": "irs_soi.ty2023.table_1_1",
            "groupby_dimension": "us:statutes/26/62#adjusted_gross_income",
            "groupby_value_id": "all",
            "measure_id": "adjusted_gross_income",
        },
    }
    fact.update(overrides)
    return fact


def _consumer_fact_row(**overrides):
    row = {
        "aggregate_fact_key": "ledger.aggregate_fact.v2:abc123",
        "legacy_fact_key": "ledger.fact.v1:abc123",
        "lineage": {
            "source_record_id": "irs_soi.ty2023.table_1_1.all.adjusted_gross_income",
            "source_cell_keys": ["ledger.source_cell.v1:cell"],
            "source_row_keys": [],
        },
        "value": 15_286_017_359_000,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {
            "level": "country",
            "id": "0100000US",
            "name": "United States",
            "vintage": "2020_census",
        },
        "entity": {"name": "tax_unit", "role": "filing_unit"},
        "observed_measure": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_measure_id": "adjusted_gross_income",
            "source_concept": "irs_soi.adjusted_gross_income",
            "unit": "usd",
        },
        "concept_alignment": {
            "source_concept": "irs_soi.adjusted_gross_income",
            "canonical_concept": "us:statutes/26/62#adjusted_gross_income",
            "relation": "exact",
            "authority": "ledger-us",
            "legal_vintage": "tax_year_2023",
        },
        "aggregation": {"method": "sum"},
        "source": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_file": "23in11si.xls",
            "url": "https://www.irs.gov/pub/irs-soi/23in11si.xls",
            "vintage": "tax_year_2023",
        },
        "dimensions": {"income_range": "all", "filing_status": "all"},
        "universe_constraints": {"domain": "all_individual_income_tax_returns"},
        "layout": {
            "record_set_id": "irs_soi.ty2023.table_1_1",
            "groupby_dimension": "us:statutes/26/62#adjusted_gross_income",
            "groupby_value_id": "all",
            "measure_id": "adjusted_gross_income",
        },
    }
    row.update(overrides)
    return row


def _consumer_fact_row_for_period(source_period: int, *, value: float):
    return _consumer_fact_row(
        aggregate_fact_key=f"ledger.aggregate_fact.v2:agi-{source_period}",
        legacy_fact_key=f"ledger.fact.v1:agi-{source_period}",
        semantic_fact_key=f"ledger.semantic_fact.v2:agi-{source_period}",
        lineage={
            "source_record_id": (
                f"irs_soi.ty{source_period}.table_1_1.all.adjusted_gross_income"
            ),
            "source_cell_keys": ["ledger.source_cell.v1:cell"],
            "source_row_keys": [],
        },
        value=value,
        period={"type": "tax_year", "value": source_period},
        source={
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_file": f"{str(source_period)[-2:]}in11si.xls",
            "url": f"https://www.irs.gov/pub/irs-soi/{str(source_period)[-2:]}in11si.xls",
            "vintage": f"tax_year_{source_period}",
        },
        layout={
            "record_set_id": f"irs_soi.ty{source_period}.table_1_1",
            "groupby_dimension": "us:statutes/26/62#adjusted_gross_income",
            "groupby_value_id": "all",
            "measure_id": "adjusted_gross_income",
        },
    )


def _monthly_consumer_fact_row(source_period: str, *, value: float):
    normalized_period = source_period.replace("-", "_")
    return _consumer_fact_row(
        aggregate_fact_key=f"ledger.aggregate_fact.v2:medicaid-{normalized_period}",
        legacy_fact_key=f"ledger.fact.v1:medicaid-{normalized_period}",
        semantic_fact_key=f"ledger.semantic_fact.v2:medicaid-{normalized_period}",
        lineage={
            "source_record_id": (
                f"cms_medicaid.month{normalized_period}."
                "us.total_medicaid_chip_enrollment"
            ),
            "source_cell_keys": ["ledger.source_cell.v1:cell"],
            "source_row_keys": [],
        },
        value=value,
        period={"type": "month", "value": source_period},
        source={
            "source_name": "cms_medicaid",
            "source_table": "Medicaid and CHIP enrollment",
            "source_file": f"enrollment_{normalized_period}.csv",
            "url": "https://data.medicaid.gov/",
            "vintage": f"month_{normalized_period}",
        },
        entity={"name": "person"},
        layout={
            "record_set_id": f"cms_medicaid.month{normalized_period}",
            "groupby_dimension": "program",
            "groupby_value_id": "total_medicaid_chip_enrollment",
            "measure_id": "total_medicaid_chip_enrollment",
        },
        observed_measure={
            "source_name": "cms_medicaid",
            "source_measure_id": "total_medicaid_chip_enrollment",
            "source_concept": "cms.total_medicaid_chip_enrollment",
            "unit": "people",
        },
    )


def _hierarchy_target(
    name: str,
    *,
    value: float,
    geography_level: str,
    geography_id: str,
    state_fips: str | None = None,
) -> TargetSpec:
    metadata = {
        "ledger_geography_level": geography_level,
        "ledger_geography_id": geography_id,
        "target_role": "population_age",
        "source_measure_id": "population",
        "target_period": "2024",
        "materializer": "population_age",
        "measure_mode": "indicator_sum",
        "age_lower_bound": "0",
        "age_upper_bound": "5",
        "age_group": "age_0_to_4",
    }
    if state_fips is not None:
        metadata["state_fips"] = state_fips
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        source="source",
        family="census_population",
        metadata=metadata,
    )


def _hierarchy_profile() -> dict[str, object]:
    return {
        "schema_version": 1,
        "hierarchy_reconciliations": [
            {
                "id": "test_cd_to_state",
                "method": "scale_children_to_parent",
                "enabled_when": {"include_congressional_district_targets": True},
                "child_geography_level": "congressional_district",
                "parent_geography_level": "state",
                "parent_geography_id": {"template": "0400000US{state_fips}"},
                "child_completeness": {
                    "child_id_metadata_key": "ledger_geography_id",
                    "parent_key_metadata_key": "state_fips",
                    "on_incomplete": "fail",
                    "expected_child_count_by_parent_key": {"01": 2},
                },
                "match": {
                    "spec_fields": ["entity", "period", "family", "filter"],
                    "metadata_keys": [
                        "target_period",
                        "target_role",
                        "source_measure_id",
                        "materializer",
                        "measure_mode",
                        "age_lower_bound",
                        "age_upper_bound",
                        "age_group",
                    ],
                },
            }
        ],
    }


def test__given_supported_ledger_fact__then_microcosm_target_preserves_lineage() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        entity_by_ledger_entity={"tax_unit": "tax_unit"},
        family_by_source_name={"irs_soi": "irs_soi"},
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_ledger_fact()], mapping)
    registry = selection.to_registry(country="us")

    # Then
    assert not selection.unsupported
    assert len(registry) == 1
    spec = registry.specs[0]
    assert spec.name == "ledger.fact.v1:abc123"
    assert spec.measure == "adjusted_gross_income"
    assert spec.filter == "is_tax_return"
    assert spec.family == "irs_soi"
    assert spec.metadata["ledger_source"] == "policyengine-ledger-data"
    assert (
        spec.metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )
    assert spec.metadata["ledger_filter_income_range"] == "all"
    assert spec.metadata["ledger_layout_record_set_id"] == "irs_soi.ty2023.table_1_1"


def test__given_consumer_contract_row__then_microcosm_target_preserves_lineage() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        entity_by_ledger_entity={"tax_unit": "tax_unit"},
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_consumer_fact_row()], mapping)

    # Then
    assert not selection.unsupported
    spec = selection.specs[0]
    assert spec.name == "ledger.aggregate_fact.v2:abc123"
    assert spec.measure == "adjusted_gross_income"
    assert (
        spec.metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )
    assert spec.metadata["ledger_fact_key"] == "ledger.aggregate_fact.v2:abc123"
    assert spec.metadata["ledger_source_concept"] == "irs_soi.adjusted_gross_income"


def test__given_consumer_contract_jsonl__then_microcosm_selects_targets(
    tmp_path,
) -> None:
    # Given
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text(json.dumps(_consumer_fact_row()) + "\n")
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        entity_by_ledger_entity={"tax_unit": "tax_unit"},
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets_from_jsonl(path, mapping)

    # Then
    assert not selection.unsupported
    assert selection.specs[0].name == "ledger.aggregate_fact.v2:abc123"


def test__given_ledger_target_reference__then_it_compiles_model_mapping() -> None:
    # Given
    reference = LedgerTargetReference(
        name="nation/irs/adjusted gross income/total",
        ledger_fact_key="ledger.aggregate_fact.v2:abc123",
        entity="tax_unit",
        measure="adjusted_gross_income",
        filter="is_tax_return",
        period=2024,
        source="IRS SOI Table 1.1",
        family="irs_soi",
        metadata={"target_role": "soi_fiscal_distribution"},
        uprating_index="cpi_u",
        uprating_from_period=2023,
        uprating_to_period=2024,
    )

    # When
    registry = compile_ledger_target_references(
        [_consumer_fact_row()],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.name == "nation/irs/adjusted gross income/total"
    assert spec.entity == "tax_unit"
    assert spec.measure == "adjusted_gross_income"
    assert spec.filter == "is_tax_return"
    assert spec.period == 2024
    assert spec.source.startswith("irs_soi | Publication 1304 Table 1.1")
    assert spec.metadata["target_role"] == "soi_fiscal_distribution"
    assert spec.metadata["uprating_index"] == "cpi_u"
    assert spec.metadata["uprating_from_period"] == "2023"
    assert spec.metadata["uprating_to_period"] == "2024"


def test__given_mean_fact_without_time_mean_contract__then_compilation_fails() -> None:
    # Given a mean-aggregated fact whose Microcosm mapping does NOT declare the
    # fact_aggregation=time_mean contract
    reference = LedgerTargetReference(
        name="usda_snap.fy2024.national_average_monthly_persons.national_total",
        ledger_fact_key="ledger.aggregate_fact.v2:abc123",
        entity="household",
        measure="snap_person_indicator",
        period=2024,
        source="USDA FNS",
        family="usda_snap",
    )

    # When / Then
    with pytest.raises(ValueError, match="unsupported aggregation 'mean'"):
        compile_ledger_target_references(
            [
                _consumer_fact_row(
                    value=41_700_000,
                    aggregation={"method": "mean"},
                    observed_measure={
                        "source_name": "usda_snap",
                        "source_table": "SNAP participation summary",
                        "source_measure_id": "average_monthly_persons",
                        "source_concept": "usda_snap.average_monthly_persons",
                        "unit": "count",
                    },
                )
            ],
            [reference],
            country="us",
        )


def test__given_count_ledger_target_reference__then_compilation_fails() -> None:
    # Given
    reference = LedgerTargetReference(
        name="census_pep.cy2024.national_resident_population_age.0_to_4.population",
        ledger_fact_key="ledger.aggregate_fact.v2:abc123",
        entity="person",
        measure="person_count",
        filter="age_0_to_4",
        period=2024,
        source="Census PEP",
        family="census_population",
    )

    # When / Then
    with pytest.raises(ValueError, match="unsupported aggregation 'count'"):
        compile_ledger_target_references(
            [
                _consumer_fact_row(
                    value=18_000_000,
                    aggregation={"method": "count"},
                    observed_measure={
                        "source_name": "census_pep",
                        "source_table": ("Annual Estimates of the Resident Population"),
                        "source_measure_id": "population",
                        "source_concept": "census_pep.resident_population",
                        "unit": "count",
                    },
                )
            ],
            [reference],
            country="us",
        )


def test__given_duplicate_semantic_facts__then_aggregate_reference_still_compiles() -> (
    None
):
    # Given
    first = _consumer_fact_row(semantic_fact_key="ledger.semantic_fact.v2:shared")
    second = _consumer_fact_row(
        aggregate_fact_key="ledger.aggregate_fact.v2:def456",
        legacy_fact_key="ledger.fact.v1:def456",
        semantic_fact_key="ledger.semantic_fact.v2:shared",
        lineage={
            "source_record_id": "irs_soi.ty2023.table_1_1.all.total_tax",
            "source_cell_keys": ["ledger.source_cell.v1:cell2"],
            "source_row_keys": [],
        },
        value=2_000_000_000_000,
        observed_measure={
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_measure_id": "total_tax",
            "source_concept": "irs_soi.total_income_tax",
            "unit": "usd",
        },
        concept_alignment={
            "source_concept": "irs_soi.total_income_tax",
            "canonical_concept": "us:statutes/26/1#income_tax",
            "relation": "exact",
            "authority": "ledger-us",
            "legal_vintage": "tax_year_2023",
        },
        layout={
            "record_set_id": "irs_soi.ty2023.table_1_1",
            "groupby_dimension": "us:statutes/26/62#adjusted_gross_income",
            "groupby_value_id": "all",
            "measure_id": "total_tax",
        },
    )
    reference = LedgerTargetReference(
        name="nation/irs/total tax/total",
        ledger_fact_key="ledger.aggregate_fact.v2:def456",
        entity="tax_unit",
        measure="income_tax",
        period=2024,
        family="irs_soi",
    )

    # When
    registry = compile_ledger_target_references(
        [first, second],
        [reference],
        country="us",
    )

    # Then
    assert registry.specs[0].value == 2_000_000_000_000
    assert (
        registry.specs[0].metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.total_tax"
    )


def test__given_ambiguous_ledger_reference_identifier__then_compilation_fails() -> None:
    # Given
    first = _consumer_fact_row(semantic_fact_key="ledger.semantic_fact.v2:shared")
    second = _consumer_fact_row(
        aggregate_fact_key="ledger.aggregate_fact.v2:def456",
        legacy_fact_key="ledger.fact.v1:def456",
        semantic_fact_key="ledger.semantic_fact.v2:shared",
        lineage={
            "source_record_id": "irs_soi.ty2023.table_1_1.all.total_tax",
            "source_cell_keys": ["ledger.source_cell.v1:cell2"],
            "source_row_keys": [],
        },
    )
    reference = LedgerTargetReference(
        name="ambiguous semantic reference",
        ledger_fact_key="ledger.semantic_fact.v2:shared",
        entity="tax_unit",
        measure="adjusted_gross_income",
    )

    # When / Then
    with pytest.raises(ValueError, match="matched multiple Ledger facts"):
        compile_ledger_target_references(
            [first, second],
            [reference],
            country="us",
        )


def test__given_missing_ledger_reference_fact__then_compilation_fails() -> None:
    # Given
    reference = LedgerTargetReference(
        name="missing fact target",
        ledger_fact_key="ledger.aggregate_fact.v2:missing",
        entity="tax_unit",
        measure="adjusted_gross_income",
    )

    # When / Then
    with pytest.raises(ValueError, match="did not match a Ledger fact identifier"):
        compile_ledger_target_references(
            [_consumer_fact_row()], [reference], country="us"
        )


def test__given_selector_matches_multiple_years__then_latest_source_period_is_used() -> (
    None
):
    # Given
    reference = LedgerTargetReference(
        name="latest SOI AGI total",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    # When
    registry = compile_ledger_target_references(
        [
            _consumer_fact_row_for_period(2022, value=14_000_000_000_000),
            _consumer_fact_row_for_period(2023, value=15_000_000_000_000),
        ],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.value == 15_000_000_000_000
    assert (
        spec.metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )


def test__given_selector_matches_future_year__then_latest_eligible_period_is_used() -> (
    None
):
    # Given
    reference = LedgerTargetReference(
        name="latest eligible SOI AGI total",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    # When
    registry = compile_ledger_target_references(
        [
            _consumer_fact_row_for_period(2023, value=15_000_000_000_000),
            _consumer_fact_row_for_period(2025, value=17_000_000_000_000),
        ],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.value == 15_000_000_000_000
    assert (
        spec.metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )


def test__given_selector_matches_future_month__then_latest_eligible_period_is_used() -> (
    None
):
    # Given
    reference = LedgerTargetReference(
        name="latest eligible Medicaid enrollment",
        ledger_selector={
            "source_name": "cms_medicaid",
            "source_measure_id": "total_medicaid_chip_enrollment",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "person",
            "layout_groupby_value_id": "total_medicaid_chip_enrollment",
        },
        entity="person",
        measure="total_medicaid_chip_enrollment",
        period=2024,
        family="cms_medicaid",
    )

    # When
    registry = compile_ledger_target_references(
        [
            _monthly_consumer_fact_row("2024-12", value=80_000_000),
            _monthly_consumer_fact_row("2025-12", value=82_000_000),
        ],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.value == 80_000_000
    assert (
        spec.metadata["ledger_source_record_id"]
        == "cms_medicaid.month2024_12.us.total_medicaid_chip_enrollment"
    )


def test__given_selector_matches_multiple_eligible_months__then_latest_month_is_used() -> (
    None
):
    # Given
    reference = LedgerTargetReference(
        name="latest eligible Medicaid enrollment",
        ledger_selector={
            "source_name": "cms_medicaid",
            "source_measure_id": "total_medicaid_chip_enrollment",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "person",
            "layout_groupby_value_id": "total_medicaid_chip_enrollment",
        },
        entity="person",
        measure="total_medicaid_chip_enrollment",
        period=2024,
        family="cms_medicaid",
    )

    # When
    registry = compile_ledger_target_references(
        [
            _monthly_consumer_fact_row("2024-11", value=79_000_000),
            _monthly_consumer_fact_row("2024-12", value=80_000_000),
        ],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.value == 80_000_000
    assert (
        spec.metadata["ledger_source_record_id"]
        == "cms_medicaid.month2024_12.us.total_medicaid_chip_enrollment"
    )


class TestLedgerSelectorExtensions:
    def test_dimension_values_use_strict_typed_equality(self) -> None:
        reference = LedgerTargetReference(
            name="typed dimension selector",
            ledger_selector={"dimension_values": {"band": "5"}},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        with pytest.raises(ValueError, match="did not match a Ledger fact selector"):
            compile_ledger_target_references(
                [_consumer_fact_row(dimensions={"band": 5})],
                [reference],
                country="us",
            )

    def test_dimension_values_accept_list_membership(self) -> None:
        reference = LedgerTargetReference(
            name="list dimension selector",
            ledger_selector={"dimension_values": {"band": ["A", "B"]}},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references(
            [_consumer_fact_row(dimensions={"band": "B"})],
            [reference],
            country="us",
        )

        assert registry.specs[0].name == "list dimension selector"

    def test_dimension_values_require_present_dimension(self) -> None:
        reference = LedgerTargetReference(
            name="missing dimension selector",
            ledger_selector={"dimension_values": {"band": "A"}},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        with pytest.raises(ValueError, match="did not match a Ledger fact selector"):
            compile_ledger_target_references(
                [_consumer_fact_row(dimensions={"income_range": "all"})],
                [reference],
                country="us",
            )

    def test_empty_membership_list_raises_instead_of_matching_nothing(self) -> None:
        reference = LedgerTargetReference(
            name="empty membership selector",
            ledger_selector={"source_measure_id": []},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        with pytest.raises(ValueError, match="empty list"):
            compile_ledger_target_references(
                [_consumer_fact_row()],
                [reference],
                country="us",
            )

    def test_empty_dimension_values_pin_list_raises(self) -> None:
        reference = LedgerTargetReference(
            name="empty pin list selector",
            ledger_selector={"dimension_values": {"band": []}},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        with pytest.raises(ValueError, match="empty pin"):
            compile_ledger_target_references(
                [_consumer_fact_row(dimensions={"band": "A"})],
                [reference],
                country="us",
            )

    def test_empty_dimensions_list_matches_dimensionless_fact(self) -> None:
        reference = LedgerTargetReference(
            name="dimensionless selector",
            ledger_selector={"dimensions": []},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references(
            [_consumer_fact_row(dimensions={})],
            [reference],
            country="us",
        )

        assert registry.specs[0].name == "dimensionless selector"

    def test_dimensions_list_matches_exact_name_set_order_insensitively(self) -> None:
        fact = _consumer_fact_row(dimensions={"band": "A", "sex": "all"})
        reference = LedgerTargetReference(
            name="dimension-name selector",
            ledger_selector={"dimensions": ["sex", "band"]},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )
        missing_name = LedgerTargetReference(
            name="partial dimension-name selector",
            ledger_selector={"dimensions": ["band"]},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references([fact], [reference], country="us")
        with pytest.raises(ValueError, match="did not match a Ledger fact selector"):
            compile_ledger_target_references([fact], [missing_name], country="us")

        assert registry.specs[0].name == "dimension-name selector"

    def test_chronicle_layout_aliases_match_fact_layout(self) -> None:
        reference = LedgerTargetReference(
            name="layout alias selector",
            ledger_selector={
                "record_set_id": "irs_soi.ty2023.table_1_1",
                "groupby_dimension": "us:statutes/26/62#adjusted_gross_income",
            },
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references(
            [_consumer_fact_row()],
            [reference],
            country="us",
        )

        assert registry.specs[0].name == "layout alias selector"

    def test_scalar_selector_accepts_list_expected_values(self) -> None:
        reference = LedgerTargetReference(
            name="list source measure selector",
            ledger_selector={
                "source_measure_id": ["total_tax", "adjusted_gross_income"],
            },
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references(
            [_consumer_fact_row()],
            [reference],
            country="us",
        )

        assert registry.specs[0].name == "list source measure selector"

    def test_us_mapping_dimensions_keep_subset_semantics(self) -> None:
        reference = LedgerTargetReference(
            name="mapping dimensions selector",
            ledger_selector={"dimensions": {"income_range": "all"}},
            entity="tax_unit",
            measure="adjusted_gross_income",
            period=2023,
            family="irs_soi",
        )

        registry = compile_ledger_target_references(
            [_consumer_fact_row()],
            [reference],
            country="us",
        )

        assert registry.specs[0].name == "mapping dimensions selector"


def test__given_selector_matches_only_future_year__then_compilation_fails() -> None:
    # Given
    reference = LedgerTargetReference(
        name="latest eligible SOI AGI total",
        ledger_selector={
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "geography_level": "country",
            "geography_id": "0100000US",
            "entity_name": "tax_unit",
            "layout_groupby_value_id": "all",
        },
        entity="tax_unit",
        measure="adjusted_gross_income",
        period=2024,
        family="irs_soi",
    )

    # When / Then
    with pytest.raises(ValueError, match="at or before target period"):
        compile_ledger_target_references(
            [_consumer_fact_row_for_period(2025, value=17_000_000_000_000)],
            [reference],
            country="us",
        )


def test__given_aggregation_method_selector__then_compilation_fails() -> None:
    # Given
    reference = LedgerTargetReference(
        name="no aggregation selector",
        ledger_selector={"aggregation_method": "sum"},
        entity="tax_unit",
        measure="adjusted_gross_income",
    )

    # When / Then
    with pytest.raises(
        ValueError,
        match="Unsupported Ledger fact selector field 'aggregation_method'",
    ):
        compile_ledger_target_references(
            [_consumer_fact_row()],
            [reference],
            country="us",
        )


@pytest.mark.parametrize("measure", [None, ""])
def test__given_non_count_reference_without_measure__then_compilation_fails(
    measure,
) -> None:
    # Given
    reference = LedgerTargetReference(
        name="missing measure target",
        ledger_fact_key="ledger.aggregate_fact.v2:abc123",
        entity="tax_unit",
        measure=measure,
    )

    # When / Then
    with pytest.raises(ValueError, match="measure is required"):
        compile_ledger_target_references(
            [_consumer_fact_row()], [reference], country="us"
        )


def test__given_reference_identifiers_match_different_facts__then_compilation_fails() -> (
    None
):
    # Given
    first = _consumer_fact_row()
    second = _consumer_fact_row(
        aggregate_fact_key="ledger.aggregate_fact.v2:def456",
        legacy_fact_key="ledger.fact.v1:def456",
        lineage={"source_record_id": "irs_soi.ty2023.table_1_1.all.total_tax"},
    )
    reference = LedgerTargetReference(
        name="inconsistent reference",
        ledger_fact_key="ledger.aggregate_fact.v2:abc123",
        ledger_source_record_id="irs_soi.ty2023.table_1_1.all.total_tax",
        entity="tax_unit",
        measure="adjusted_gross_income",
    )

    # When / Then
    with pytest.raises(ValueError, match="resolve to different Ledger facts"):
        compile_ledger_target_references([first, second], [reference], country="us")


def test__given_lineage_only_metadata_diff__then_parity_report_names_full_digest() -> (
    None
):
    # Given
    expected = TargetRegistry(
        [
            TargetSpec(
                name="agi",
                entity="tax_unit",
                measure="adjusted_gross_income",
                value=100.0,
                source="source",
            )
        ],
        country="us",
    )
    actual = TargetRegistry(
        [
            TargetSpec(
                name="agi",
                entity="tax_unit",
                measure="adjusted_gross_income",
                value=100.0,
                source="source",
                metadata={"ledger_fact_key": "fact"},
            )
        ],
        country="us",
    )

    # When
    report = ledger_target_registry_parity_report(expected, actual)

    # Then
    assert not report.passed
    assert any(
        "full target registry digest differs" in line for line in report.failures
    )
    assert (
        report.details["expected_calibration_digest"]
        == report.details["actual_calibration_digest"]
    )
    assert asdict(expected.specs[0]) != asdict(actual.specs[0])


def test__given_domain_scoped_fact_without_filter_mapping__then_it_is_unsupported() -> (
    None
):
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        }
    )

    # When
    selection = select_ledger_targets([_ledger_fact()], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "missing_model_filter_mapping"


def test__given_domain_scoped_fact_with_domain_filter__then_target_uses_filter() -> (
    None
):
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_ledger_fact()], mapping)

    # Then
    assert not selection.unsupported
    assert selection.specs[0].filter == "is_tax_return"


def test__given_scoped_fact_without_filter_mapping__then_it_is_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        }
    )
    fact = _ledger_fact(
        source_record_id="irs_soi.ty2023.table_1_1.under_1.adjusted_gross_income",
        filters={
            "income_range": "under_1",
            "filing_status": "all",
            "agi_upper_usd": 1,
        },
        constraints=[
            {
                "variable": "us:statutes/26/62#adjusted_gross_income",
                "operator": "<",
                "value": 1,
                "unit": "usd",
                "role": "filter",
            }
        ],
    )

    # When
    selection = select_ledger_targets([fact], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "missing_model_filter_mapping"


def test__given_scoped_fact_with_only_domain_filter__then_it_is_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )
    fact = _ledger_fact(
        source_record_id="irs_soi.ty2023.table_1_1.under_1.adjusted_gross_income",
        filters={
            "income_range": "under_1",
            "filing_status": "all",
            "agi_upper_usd": 1,
        },
    )

    # When
    selection = select_ledger_targets([fact], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "missing_model_filter_mapping"


def test__given_scoped_fact_with_filter_mapping__then_target_uses_filter() -> None:
    # Given
    source_record_id = "irs_soi.ty2023.table_1_1.under_1.adjusted_gross_income"
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_source_record_id={source_record_id: "agi_under_1"},
    )
    fact = _ledger_fact(
        source_record_id=source_record_id,
        domain="",
        filters={"income_range": "under_1", "filing_status": "all"},
    )

    # When
    selection = select_ledger_targets([fact], mapping)

    # Then
    assert not selection.unsupported
    assert selection.specs[0].filter == "agi_under_1"


def test__given_unmapped_ledger_fact__then_it_is_reported_as_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping()

    # When
    selection = select_ledger_targets([_ledger_fact()], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "missing_model_measure_mapping"
    assert (
        selection.unsupported[0].source_record_id
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )


def test__given_rate_fact__then_it_is_reported_as_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )
    fact = _ledger_fact(aggregation={"method": "rate"})

    # When
    selection = select_ledger_targets([fact], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "unsupported_aggregation:rate"


def test__given_count_fact__then_it_is_reported_as_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets(
        [_ledger_fact(aggregation={"method": "count"}, value=10)],
        mapping,
    )

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "unsupported_aggregation:count"


def test__given_hierarchy_profile__then_children_scale_to_parent() -> None:
    # Given
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "al_01",
                value=30.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_02",
                value=70.0,
                geography_level="congressional_district",
                geography_id="5001900US0102",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_state",
                value=200.0,
                geography_level="state",
                geography_id="0400000US01",
                state_fips="01",
            ),
        ],
        country="us",
    )

    # When
    reconciled = apply_ledger_target_profile(
        registry,
        _hierarchy_profile(),
        context={"include_congressional_district_targets": True},
    )

    # Then
    first, second, parent = reconciled.specs
    assert first.value == pytest.approx(60.0)
    assert second.value == pytest.approx(140.0)
    assert parent.value == pytest.approx(200.0)
    assert first.metadata["hierarchy_reconciliation_rule"] == "test_cd_to_state"
    assert first.metadata["hierarchy_raw_value"] == "30"
    assert first.metadata["hierarchy_child_sum_raw"] == "100"
    assert first.metadata["hierarchy_parent_value"] == "200"
    assert first.metadata["hierarchy_coverage_ratio"] == "0.5"
    assert first.metadata["hierarchy_reconciliation_factor"] == "2"
    assert first.metadata["hierarchy_parent_geography_id"] == "0400000US01"
    assert first.metadata["hierarchy_parent_key"] == "01"
    assert first.metadata["hierarchy_expected_child_count"] == "2"
    assert first.metadata["hierarchy_observed_child_count"] == "2"
    assert first.metadata["hierarchy_child_ids"] == "5001900US0101,5001900US0102"


def test__given_disabled_hierarchy_profile__then_children_are_unchanged() -> None:
    # Given
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "al_01",
                value=30.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_state",
                value=200.0,
                geography_level="state",
                geography_id="0400000US01",
                state_fips="01",
            ),
        ],
        country="us",
    )

    # When
    reconciled = apply_ledger_target_profile(
        registry,
        _hierarchy_profile(),
        context={"include_congressional_district_targets": False},
    )

    # Then
    assert reconciled.specs[0].value == pytest.approx(30.0)
    assert "hierarchy_reconciliation_rule" not in reconciled.specs[0].metadata


def test__given_nonzero_parent_and_zero_children__then_hierarchy_fails() -> None:
    # Given
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "al_01",
                value=0.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_02",
                value=0.0,
                geography_level="congressional_district",
                geography_id="5001900US0102",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_state",
                value=200.0,
                geography_level="state",
                geography_id="0400000US01",
                state_fips="01",
            ),
        ],
        country="us",
    )

    # When / Then
    with pytest.raises(ValueError, match="zero-valued children to nonzero parent"):
        apply_ledger_target_profile(
            registry,
            _hierarchy_profile(),
            context={"include_congressional_district_targets": True},
        )


def test__given_incomplete_hierarchy_children__then_reconciliation_fails() -> None:
    # Given
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "al_01",
                value=30.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_state",
                value=200.0,
                geography_level="state",
                geography_id="0400000US01",
                state_fips="01",
            ),
        ],
        country="us",
    )

    # When / Then
    with pytest.raises(ValueError, match="expected 2 child target"):
        apply_ledger_target_profile(
            registry,
            _hierarchy_profile(),
            context={"include_congressional_district_targets": True},
        )


def test__given_duplicate_hierarchy_child_ids__then_reconciliation_fails() -> None:
    # Given
    registry = TargetRegistry(
        [
            _hierarchy_target(
                "al_01a",
                value=30.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_01b",
                value=70.0,
                geography_level="congressional_district",
                geography_id="5001900US0101",
                state_fips="01",
            ),
            _hierarchy_target(
                "al_state",
                value=200.0,
                geography_level="state",
                geography_id="0400000US01",
                state_fips="01",
            ),
        ],
        country="us",
    )

    # When / Then
    with pytest.raises(ValueError, match="duplicate child ids"):
        apply_ledger_target_profile(
            registry,
            _hierarchy_profile(),
            context={"include_congressional_district_targets": True},
        )


def test__given_malformed_value__then_it_is_reported_as_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_ledger_fact(value="not numeric")], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "invalid_value"


def test__given_overflow_value__then_it_is_reported_as_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_ledger_fact(value="1e10000")], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "invalid_value"


def test__given_negative_value_without_signed_mapping__then_it_is_unsupported() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    selection = select_ledger_targets([_ledger_fact(value=-1)], mapping)

    # Then
    assert not selection.specs
    assert selection.unsupported[0].reason == "missing_signed_target_mapping"


def test__given_negative_value_with_signed_mapping__then_target_is_signed() -> None:
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
        signed_by_concept=frozenset({"us:statutes/26/62#adjusted_gross_income"}),
    )

    # When
    selection = select_ledger_targets([_ledger_fact(value=-1)], mapping)

    # Then
    assert not selection.unsupported
    assert selection.specs[0].signed is True


def test__given_ledger_target_metadata__then_coverage_gate_uses_structured_fields() -> (
    None
):
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )
    selection = select_ledger_targets([_ledger_fact()], mapping)
    requirement = TargetCoverageRequirement(
        requirement_id="soi_agi_total",
        label="SOI AGI total",
        accepted_measures=("adjusted_gross_income",),
        required_metadata=(
            ("ledger_source", "policyengine-ledger-data"),
            ("ledger_measure_concept", "us:statutes/26/62#adjusted_gross_income"),
            ("ledger_geography_level", "country"),
            ("ledger_filter_income_range", "all"),
        ),
    )

    # When
    result = target_profile_coverage_gate(selection.specs, [requirement])

    # Then
    assert result.passed


def test__given_current_soi_like_row__then_ledger_adapter_matches_current_target_shape() -> (
    None
):
    # Given
    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        entity_by_ledger_entity={"tax_unit": "tax_unit"},
        family_by_source_name={"irs_soi": "irs_soi"},
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_return"},
    )

    # When
    spec = select_ledger_targets([_ledger_fact()], mapping).specs[0]

    # Then
    assert spec.entity == "tax_unit"
    assert spec.measure == "adjusted_gross_income"
    assert spec.value == 15_286_017_359_000
    assert spec.period == 2023
    assert spec.family == "irs_soi"
    assert spec.signed is False


def test_ledger_metadata_records_assertion_and_fact_period():
    from microcosm.build.ledger_targets import (
        LedgerTargetMapping,
        target_spec_from_ledger_fact,
    )

    mapping = LedgerTargetMapping(
        measure_by_concept={
            "us:statutes/26/62#adjusted_gross_income": "adjusted_gross_income"
        },
        filter_by_domain={"all_individual_income_tax_returns": "is_tax_filer"},
    )
    # Legacy rows that omit the assertion field are not stamped in metadata
    # (absence means observation-by-default to readers, matching the loader).
    legacy = target_spec_from_ledger_fact(_consumer_fact_row(), mapping)
    assert "ledger_assertion" not in legacy.metadata
    assert legacy.metadata["ledger_fact_period"] == "2023"

    observation = target_spec_from_ledger_fact(
        _consumer_fact_row(assertion="observation"),
        mapping,
    )
    assert observation.metadata["ledger_assertion"] == "observation"

    projection = target_spec_from_ledger_fact(
        _consumer_fact_row(assertion="source_projection"),
        mapping,
    )
    assert projection.metadata["ledger_assertion"] == "source_projection"


def test_ledger_reference_selector_matches_on_assertion():
    from microcosm.build.ledger_targets import (
        LedgerTargetReference,
        compile_ledger_target_references,
    )

    rows = [
        _consumer_fact_row(),
        _consumer_fact_row(
            aggregate_fact_key="ledger.aggregate_fact.v2:proj123",
            legacy_fact_key="ledger.fact.v1:proj123",
            assertion="source_projection",
            value=16_000_000_000_000,
            lineage={
                "source_record_id": "cbo.ty2023.projection.agi",
                "source_cell_keys": ["ledger.source_cell.v1:proj"],
                "source_row_keys": [],
            },
        ),
    ]
    reference = LedgerTargetReference(
        name="cbo_projected_agi",
        ledger_selector={
            "source_measure_id": "adjusted_gross_income",
            "assertion": "source_projection",
        },
        entity="household",
        measure="adjusted_gross_income",
        family="cbo",
    )

    registry = compile_ledger_target_references(rows, [reference], country="us")
    (spec,) = registry.specs
    assert spec.value == 16_000_000_000_000
    assert spec.metadata["ledger_assertion"] == "source_projection"
