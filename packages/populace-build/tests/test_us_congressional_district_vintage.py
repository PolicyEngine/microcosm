"""US congressional-district geography-vintage translation tests."""

import pytest

from populace.build.us_runtime import (
    SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID,
    compile_us_fiscal_target_registry,
    congressional_district_distribution_from_ledger_facts,
    load_congressional_district_vintage_crosswalk,
    translate_congressional_district_facts_to_current_vintage,
)
from populace.build.us_runtime.fiscal_targets import US_FISCAL_TARGET_REFERENCES


def test_translate_cd_facts_conserves_values_and_records_lineage() -> None:
    facts = [
        _soi_cd_fact(
            "adjusted_gross_income",
            100.0,
            geography_id="5001700US0601",
            source_row_id="ca_01",
        ),
        _soi_cd_fact(
            "adjusted_gross_income",
            60.0,
            geography_id="5001700US0653",
            source_row_id="ca_53",
        ),
    ]
    crosswalk = [
        {
            "source_geography_id": "5001700US0601",
            "target_geography_id": "5001900US0601",
            "weight": 3.0,
        },
        {
            "source_geography_id": "5001700US0601",
            "target_geography_id": "5001900US0602",
            "weight": 1.0,
        },
        {
            "source_geography_id": "5001700US0653",
            "target_geography_id": "5001900US0602",
            "weight": 2.0,
        },
    ]

    translated = translate_congressional_district_facts_to_current_vintage(
        facts,
        crosswalk,
        crosswalk_basis="block_population",
    )

    by_geography = {fact["geography"]["id"]: fact for fact in translated}
    assert set(by_geography) == {"5001900US0601", "5001900US0602"}
    assert by_geography["5001900US0601"]["value"] == pytest.approx(75.0)
    assert by_geography["5001900US0602"]["value"] == pytest.approx(85.0)
    merged_lineage = by_geography["5001900US0602"]["lineage"]
    assert merged_lineage["target_geography_vintage"] == "118th_congress"
    assert (
        merged_lineage["congressional_district_vintage_crosswalk_basis"]
        == "block_population"
    )
    assert merged_lineage["source_record_ids"] == [
        "irs_soi.ty2023.congressional_district_2022.all_returns.ca_01.adjusted_gross_income",
        "irs_soi.ty2023.congressional_district_2022.all_returns.ca_53.adjusted_gross_income",
    ]
    assert by_geography["5001900US0602"]["lineage"]["source_record_id"].startswith(
        f"{SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID}.current_cd.0602."
    )


def test_translate_cd_facts_requires_crosswalk_for_old_vintage_rows() -> None:
    facts = [
        _soi_cd_fact(
            "return_count",
            100.0,
            geography_id="5001700US0601",
            source_row_id="ca_01",
        )
    ]

    with pytest.raises(ValueError, match="missing source geography"):
        translate_congressional_district_facts_to_current_vintage(
            facts,
            [
                {
                    "source_geography_id": "5001700US0602",
                    "target_geography_id": "5001900US0602",
                    "weight": 1.0,
                }
            ],
        )


def test_translate_cd_facts_rejects_cross_state_crosswalk_rows() -> None:
    facts = [
        _soi_cd_fact(
            "return_count",
            100.0,
            geography_id="5001700US0601",
            source_row_id="ca_01",
        )
    ]

    with pytest.raises(ValueError, match="must stay within the source state"):
        translate_congressional_district_facts_to_current_vintage(
            facts,
            [
                {
                    "source_geography_id": "5001700US0601",
                    "target_geography_id": "5001900US0801",
                    "weight": 1.0,
                }
            ],
        )


def test_translate_cd_facts_rejects_invalid_source_values() -> None:
    fact = _soi_cd_fact(
        "return_count",
        100.0,
        geography_id="5001700US0601",
        source_row_id="ca_01",
    )
    fact["value"] = "not-a-number"

    with pytest.raises(ValueError, match="has invalid value"):
        translate_congressional_district_facts_to_current_vintage(
            [fact],
            [
                {
                    "source_geography_id": "5001700US0601",
                    "target_geography_id": "5001900US0601",
                    "weight": 1.0,
                }
            ],
        )


def test_load_cd_vintage_crosswalk_from_csv_normalizes_weights(tmp_path) -> None:
    path = tmp_path / "cd_crosswalk.csv"
    path.write_text(
        "source_geography_id,target_geography_id,weight\n"
        "5001700US0807,5001900US0807,2\n"
        "5001700US0807,5001900US0808,1\n"
    )

    crosswalk = load_congressional_district_vintage_crosswalk(path)

    assert crosswalk["share"].tolist() == pytest.approx([2.0 / 3.0, 1.0 / 3.0])


def test_compile_us_fiscal_targets_can_use_translated_current_cd_surface() -> None:
    facts = [
        *_packaged_reference_facts(),
        _soi_cd_fact(
            "adjusted_gross_income",
            100.0,
            geography_id="5001700US0601",
            source_row_id="ca_01",
        ),
        _soi_cd_fact(
            "adjusted_gross_income",
            60.0,
            geography_id="5001700US0653",
            source_row_id="ca_53",
        ),
    ]
    crosswalk = [
        {
            "source_geography_id": "5001700US0601",
            "target_geography_id": "5001900US0601",
            "weight": 3.0,
        },
        {
            "source_geography_id": "5001700US0601",
            "target_geography_id": "5001900US0602",
            "weight": 1.0,
        },
        {
            "source_geography_id": "5001700US0653",
            "target_geography_id": "5001900US0602",
            "weight": 2.0,
        },
    ]

    registry = compile_us_fiscal_target_registry(
        facts,
        include_congressional_district_targets=True,
        congressional_district_vintage_crosswalk=crosswalk,
    )

    cd_specs = {
        spec.metadata["congressional_district_geoid"]: spec
        for spec in registry.specs
        if spec.metadata.get("source_measure_id") == "adjusted_gross_income"
        and spec.metadata.get("ledger_geography_level") == "congressional_district"
    }
    assert set(cd_specs) == {"0601", "0602"}
    assert cd_specs["0601"].metadata["ledger_geography_id"] == "5001900US0601"
    assert cd_specs["0601"].metadata["ledger_geography_vintage"] == "118th_congress"
    assert "hierarchy_reconciliation_factor" not in cd_specs["0601"].metadata
    assert cd_specs["0601"].value == pytest.approx(75.0)
    assert cd_specs["0602"].value == pytest.approx(85.0)


def test_cd_distribution_accepts_translated_current_cd_facts() -> None:
    translated = translate_congressional_district_facts_to_current_vintage(
        [
            _soi_cd_fact(
                "return_count",
                100.0,
                geography_id="5001700US0807",
                source_row_id="co_07",
            )
        ],
        [
            {
                "source_geography_id": "5001700US0807",
                "target_geography_id": "5001900US0807",
                "weight": 2.0,
            },
            {
                "source_geography_id": "5001700US0807",
                "target_geography_id": "5001900US0808",
                "weight": 1.0,
            },
        ],
    )

    distribution = congressional_district_distribution_from_ledger_facts(translated)

    assert distribution["congressional_district_geoid"].tolist() == ["0807", "0808"]
    assert distribution["sampling_weight"].tolist() == pytest.approx(
        [200.0 / 3.0, 100.0 / 3.0]
    )


def _soi_cd_fact(
    measure_id: str,
    value: float,
    *,
    geography_id: str,
    source_row_id: str,
    geography_level: str = "congressional_district",
) -> dict[str, object]:
    source_record_id = (
        f"{SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID}.{source_row_id}.{measure_id}"
    )
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{source_row_id}.{measure_id}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{source_row_id}.{measure_id}",
        "legacy_fact_key": f"ledger.fact.v1:{source_row_id}.{measure_id}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {
            "level": geography_level,
            "id": geography_id,
            "vintage": "117th_congress",
        },
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "dimensions": {"income_range": "all", "filing_status": "all"},
        "layout": {
            "record_set_id": SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID,
            "groupby_dimension": "irs_soi.congressional_district",
            "groupby_value_id": source_row_id,
            "measure_id": measure_id,
            "source_row_id": source_row_id,
        },
        "observed_measure": {
            "source_name": "irs_soi",
            "source_table": "SOI congressional district returns",
            "source_measure_id": measure_id,
            "source_concept": measure_id,
            "unit": "usd",
        },
        "source": {
            "source_name": "irs_soi",
            "source_table": "SOI congressional district returns",
            "source_sha256": "sha",
            "vintage": "tax_year_2023",
            "url": "https://example.org/soi-cd",
        },
    }


def _packaged_reference_facts() -> list[dict[str, object]]:
    return [
        _ledger_fact_for_reference(reference, value=index + 1)
        for index, reference in enumerate(US_FISCAL_TARGET_REFERENCES)
    ]


def _ledger_fact_for_reference(reference, *, value: float) -> dict[str, object]:
    source_record_id = reference.ledger_source_record_id or reference.name
    return {
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "tax_year", "value": reference.period},
        "entity": {"name": reference.entity},
        "aggregation": {"method": "sum"},
        "geography": {"level": "country", "id": "0100000US"},
        "dimensions": {},
        "layout": {
            "record_set_id": f"{reference.family}.record_set",
            "groupby_dimension": "",
            "groupby_value_id": "all",
            "measure_id": reference.measure or reference.name,
        },
        "observed_measure": {
            "source_name": reference.family,
            "source_table": f"{reference.family} table",
            "source_measure_id": reference.measure or reference.name,
            "source_concept": reference.measure or reference.name,
            "unit": "usd",
        },
        "source": {
            "source_name": reference.family,
            "source_table": f"{reference.family} table",
            "vintage": str(reference.period),
            "url": "https://example.org/reference",
        },
    }
