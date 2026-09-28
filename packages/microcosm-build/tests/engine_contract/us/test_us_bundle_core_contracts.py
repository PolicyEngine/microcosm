"""Tests split from packages/microcosm-build/tests/test_us_bundle_core_contracts.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_bundle_core_contracts import *


def test_us_catalog_has_complete_explicit_contracts() -> None:
    __import__("policyengine_us")
    catalog = build_catalogs()
    columns = catalog["columns"]
    assert len(columns) == 176
    assert len({row["key"] for row in columns}) == 176
    assert catalog["metadata_waivers"] == [
        {
            "id": "policyengine_us_unit_unavailable",
            "field": "unit",
            "authority": "PolicyEngineUSVariableMetadataIndex",
            "public": False,
            "expires_on": "2026-11-16",
            "reason": (
                "The import-free installed-engine index exposes entity, dtype, "
                "and period but no physical unit. Numeric columns remain "
                "internal until a reviewed unit authority lands."
            ),
        }
    ]
    for row in columns:
        contract = row["contract"]
        assert row["key"].startswith(f"{contract['entity']}.")
        assert contract["domain"]
        assert contract["public_stability"] == "internal"
        if contract["unit"] == "unit_not_declared_by_engine_metadata":
            assert contract["unit_waiver"] == "policyengine_us_unit_unavailable"
        else:
            assert "unit_waiver" not in contract
    assert Counter(row["contract"]["unit"] for row in columns) == {
        "unit_not_declared_by_engine_metadata": 89,
        "boolean": 51,
        "count": 27,
        "categorical": 9,
    }
    assert {
        row["key"] for row in columns if row["key"].endswith(".@resolved_weight")
    } == {
        f"{entity}.@resolved_weight"
        for entity in (
            "family",
            "household",
            "marital_unit",
            "person",
            "spm_unit",
            "tax_unit",
        )
    }
    load_schema_registry().validate(catalog, "catalogs.schema.json")

    resolved = _resolve_vintages(
        {
            **_vintage_resources(),
            "catalogs": catalog,
        }
    )
    assert len(resolved.columns) == 176
    by_key = {row["key"]: row["contract"] for row in columns}
    for column in resolved.columns:
        contract = by_key[column.key]
        assert column.entity.id == contract["entity"]
        assert column.dtype == contract["dtype"]
        assert column.unit == contract["unit"]
        assert column.period == contract["definition_period"]
        assert column.vintage == contract["vintage"]
        assert column.nullable is contract["nullable"]
        assert column.domain == contract["domain"]
        assert column.public_stability == contract["public_stability"]
        assert column.unit_waiver == contract.get("unit_waiver")


@pytest.mark.parametrize("field", ["unit", "domain", "public_stability"])
def test_catalog_schema_requires_complete_contract_fields(field: str) -> None:
    __import__("policyengine_us")
    catalog = build_catalogs()
    del catalog["columns"][0]["contract"][field]
    with pytest.raises(SpecValidationError, match=field):
        load_schema_registry().validate(catalog, "catalogs.schema.json")
