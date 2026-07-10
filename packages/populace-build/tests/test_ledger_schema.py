"""Tests for the vendored stdlib consumer-fact schema validator."""

from __future__ import annotations

import hashlib

import pytest

from populace.build.ledger_schema import (
    CONSUMER_FACT_SCHEMA_SHA256,
    validate_consumer_fact_row,
)

# 24 lowercase-hex characters: the identity-key suffix the schema pins.
_HEX = "0123456789abcdef01234567"

# Pinning the exact bytes of the vendored schema. A silent edit to
# schemas/consumer_fact.v1.schema.json (drift from the published Ledger
# contract) fails here instead of quietly changing what rows are accepted.
_EXPECTED_SCHEMA_SHA256 = (
    "a3e52d3bac4dea497a30722fb70446a79bf11614d1c5e88a34e21a4b3f27acf1"
)


def _schema_complete_fact_row(**overrides):
    """A row satisfying every required field and identity-key pattern."""
    row = {
        "schema_version": "ledger.consumer_fact.v1",
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{_HEX}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{_HEX}",
        "legacy_fact_key": f"ledger.fact.v1:{_HEX}",
        "source_release_key": f"ledger.source_release.v2:{_HEX}",
        "source_series_key": f"ledger.source_series.v2:{_HEX}",
        "observed_measure_key": f"ledger.observed_measure.v2:{_HEX}",
        "dimension_set_key": f"ledger.dimension_set.v2:{_HEX}",
        "universe_constraint_set_key": f"ledger.universe_constraint_set.v2:{_HEX}",
        "value": 15_286_017_359_000,
        "value_type": "integer",
        "assertion": "observation",
        "period": {"type": "tax_year", "value": 2023},
        "geography": {"level": "country", "id": "0100000US"},
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "observed_measure": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_measure_id": "adjusted_gross_income",
            "source_concept": "irs_soi.adjusted_gross_income",
            "unit": "usd",
        },
        "dimensions": {"income_range": "all", "filing_status": "all"},
        "universe_constraints": {"domain": "all_individual_income_tax_returns"},
        "source": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 1.1",
            "source_file": "23in11si.xls",
            "vintage": "tax_year_2023",
            "extracted_at": "2024-01-01T00:00:00Z",
            "extraction_method": "manual",
            "source_sha256": "ab" * 32,
            "source_size_bytes": 1024,
            "raw_r2_uri": "r2://ledger/irs_soi/23in11si.xls",
        },
        "lineage": {
            "source_record_id": "irs_soi.ty2023.table_1_1.all.adjusted_gross_income",
            "source_cell_keys": ["ledger.source_cell.v1:cell"],
        },
    }
    row.update(overrides)
    return row


def test_vendored_schema_sha256_is_pinned():
    assert CONSUMER_FACT_SCHEMA_SHA256 == _EXPECTED_SCHEMA_SHA256


def test_schema_sha256_matches_vendored_bytes():
    from importlib.resources import files

    raw = files("populace.build.schemas").joinpath(
        "consumer_fact.v1.schema.json"
    ).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CONSUMER_FACT_SCHEMA_SHA256


def test_schema_complete_row_validates():
    validate_consumer_fact_row(_schema_complete_fact_row(), line_number=1, source="feed")


def test_row_must_be_object():
    with pytest.raises(ValueError, match="expected a JSON object"):
        validate_consumer_fact_row([1, 2, 3], line_number=1, source="feed")


def test_missing_top_level_required_field_reports_path():
    row = _schema_complete_fact_row()
    del row["entity"]
    with pytest.raises(ValueError, match="'entity' is required but missing"):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_missing_nested_required_field_reports_path():
    row = _schema_complete_fact_row(period={"type": "tax_year"})
    with pytest.raises(ValueError, match="'period.value' is required but missing"):
        validate_consumer_fact_row(row, line_number=3, source="feed")

    row = _schema_complete_fact_row(
        observed_measure={
            "source_name": "irs_soi",
            "source_table": "t",
            "source_measure_id": "agi",
            "source_concept": "irs_soi.agi",
        }
    )
    with pytest.raises(ValueError, match="'observed_measure.unit' is required"):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_wrong_const_schema_version_rejected():
    row = _schema_complete_fact_row(schema_version="ledger.consumer_fact.v2")
    with pytest.raises(ValueError, match="expected constant 'ledger.consumer_fact.v1'"):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_wrong_enum_rejected():
    row = _schema_complete_fact_row(assertion="policyengine_aged")
    with pytest.raises(ValueError, match="is not one of"):
        validate_consumer_fact_row(row, line_number=3, source="feed")

    row = _schema_complete_fact_row(value_type="money")
    with pytest.raises(ValueError, match="'value_type'"):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_pattern_violation_reports_field_and_pattern():
    row = _schema_complete_fact_row(aggregate_fact_key="ledger.aggregate_fact.v2:XYZ")
    with pytest.raises(ValueError, match="does not match pattern"):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_extra_top_level_field_rejected():
    row = _schema_complete_fact_row(surprise_field=1)
    with pytest.raises(ValueError, match="'surprise_field' is not an allowed property"):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_extra_nested_field_rejected():
    row = _schema_complete_fact_row(
        period={"type": "tax_year", "value": 2023, "unexpected": 1}
    )
    with pytest.raises(ValueError, match="'period.unexpected' is not an allowed"):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_wrong_scalar_type_rejected():
    row = _schema_complete_fact_row(value={"nested": "object"})
    with pytest.raises(ValueError, match="'value': expected type"):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_integer_field_rejects_string_and_boolean():
    row = _schema_complete_fact_row()
    row["source"]["source_size_bytes"] = "1024"
    with pytest.raises(ValueError, match="source.source_size_bytes.*expected type integer"):
        validate_consumer_fact_row(row, line_number=3, source="feed")

    row = _schema_complete_fact_row()
    row["source"]["source_size_bytes"] = True
    with pytest.raises(ValueError, match="source.source_size_bytes.*got boolean"):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_dimensions_additional_property_type_enforced():
    row = _schema_complete_fact_row(dimensions={"income_range": {"nested": "bad"}})
    with pytest.raises(ValueError, match="'dimensions.income_range': expected type"):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_array_items_validated_via_ref():
    row = _schema_complete_fact_row(
        universe_constraints={
            "domain": "all_returns",
            "constraints": [
                {"variable": "age", "operator": ">=", "value": 0, "role": "filter"},
                {"variable": "age", "operator": "<", "value": 5},
            ],
        }
    )
    with pytest.raises(
        ValueError, match=r"universe_constraints.constraints\[1\].role' is required"
    ):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_lineage_array_item_type_enforced():
    row = _schema_complete_fact_row()
    row["lineage"]["source_cell_keys"] = [1, 2]
    with pytest.raises(
        ValueError, match=r"lineage.source_cell_keys\[0\]': expected type string"
    ):
        validate_consumer_fact_row(row, line_number=3, source="feed")


def test_sol_counterexample_non_finite_number_value_is_rejected():
    # json.loads accepts the JS constants NaN/Infinity/-Infinity and hands back
    # a float that satisfies the ``number`` type check, silently poisoning a
    # calibration target (finding #7). The validator must reject a non-finite
    # numeric wherever it appears in a row.
    for bad in (float("nan"), float("inf"), float("-inf")):
        row = _schema_complete_fact_row(value=bad)
        with pytest.raises(ValueError, match="is not finite"):
            validate_consumer_fact_row(row, line_number=3, source="feed")

    # Non-finite in a nested numeric field is caught too.
    nested = _schema_complete_fact_row()
    nested["source"]["source_size_bytes"] = float("inf")
    with pytest.raises(ValueError, match="is not finite"):
        validate_consumer_fact_row(nested, line_number=3, source="feed")
