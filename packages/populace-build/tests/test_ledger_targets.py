import json
from dataclasses import asdict

import pytest

from populace.build.gates import TargetCoverageRequirement, target_profile_coverage_gate
from populace.build.ledger_targets import (
    LedgerTargetMapping,
    LedgerTargetReference,
    compile_ledger_target_references,
    ledger_target_registry_parity_report,
    select_ledger_targets,
    select_ledger_targets_from_jsonl,
)
from populace.calibrate import TargetRegistry, TargetSpec


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


def test__given_supported_ledger_fact__then_populace_target_preserves_lineage() -> None:
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


def test__given_consumer_contract_row__then_populace_target_preserves_lineage() -> None:
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


def test__given_consumer_contract_jsonl__then_populace_selects_targets(
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


def test__given_count_ledger_target_reference__then_it_compiles_as_sum_spec() -> None:
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

    # When
    registry = compile_ledger_target_references(
        [
            _consumer_fact_row(
                value=18_000_000,
                aggregation={"method": "count"},
                observed_measure={
                    "source_name": "census_pep",
                    "source_table": "Annual Estimates of the Resident Population",
                    "source_measure_id": "population",
                    "source_concept": "census_pep.resident_population",
                    "unit": "count",
                },
            )
        ],
        [reference],
        country="us",
    )

    # Then
    spec = registry.specs[0]
    assert spec.measure == "person_count"
    assert spec.filter == "age_0_to_4"
    assert spec.value == 18_000_000
    assert spec.metadata["ledger_aggregation_method"] == "count"


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
            "aggregation_method": "sum",
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
            "aggregation_method": "sum",
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
            "aggregation_method": "sum",
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
            "aggregation_method": "sum",
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
            "aggregation_method": "sum",
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


def test__given_count_fact__then_populace_target_is_still_sum_only() -> None:
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
    assert not selection.unsupported
    spec = selection.specs[0]
    assert spec.measure == "adjusted_gross_income"
    assert spec.value == 10
    assert spec.metadata["ledger_aggregation_method"] == "count"


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
