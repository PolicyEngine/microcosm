import json

from populace.build.gates import TargetCoverageRequirement, target_profile_coverage_gate
from populace.build.ledger_targets import (
    LedgerTargetMapping,
    select_ledger_targets,
    select_ledger_targets_from_jsonl,
)


def _ledger_fact(**overrides):
    fact = {
        "fact_key": "arch.fact.v1:abc123",
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
            "concept_authority": "arch-us",
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
        "aggregate_fact_key": "arch.aggregate_fact.v2:abc123",
        "legacy_fact_key": "arch.fact.v1:abc123",
        "lineage": {
            "source_record_id": "irs_soi.ty2023.table_1_1.all.adjusted_gross_income",
            "source_cell_keys": ["arch.source_cell.v1:cell"],
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
            "authority": "arch-us",
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
    assert spec.name == "arch.fact.v1:abc123"
    assert spec.measure == "adjusted_gross_income"
    assert spec.filter == "is_tax_return"
    assert spec.family == "irs_soi"
    assert spec.metadata["ledger_source"] == "policyengine_ledger"
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
    assert spec.name == "arch.aggregate_fact.v2:abc123"
    assert spec.measure == "adjusted_gross_income"
    assert (
        spec.metadata["ledger_source_record_id"]
        == "irs_soi.ty2023.table_1_1.all.adjusted_gross_income"
    )
    assert spec.metadata["ledger_fact_key"] == "arch.aggregate_fact.v2:abc123"
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
    assert selection.specs[0].name == "arch.aggregate_fact.v2:abc123"


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
            ("ledger_source", "policyengine_ledger"),
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
    assert spec.aggregation == "sum"
    assert spec.value == 15_286_017_359_000
    assert spec.period == 2023
    assert spec.family == "irs_soi"
    assert spec.signed is False
