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
    assert merged_lineage["target_geography_vintage"] == "119th_congress"
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
        allow_unaged_dollar_targets=True,
    )

    cd_specs = {
        spec.metadata["congressional_district_geoid"]: spec
        for spec in registry.specs
        if spec.metadata.get("source_measure_id") == "adjusted_gross_income"
        and spec.metadata.get("ledger_geography_level") == "congressional_district"
    }
    assert set(cd_specs) == {"0601", "0602"}
    assert cd_specs["0601"].metadata["ledger_geography_id"] == "5001900US0601"
    assert cd_specs["0601"].metadata["ledger_geography_vintage"] == "119th_congress"
    assert "hierarchy_reconciliation_factor" not in cd_specs["0601"].metadata
    assert cd_specs["0601"].value == pytest.approx(75.0)
    assert cd_specs["0602"].value == pytest.approx(85.0)


def test__given_missing_statewide_source_cd__then_matching_state_fact_is_split_to_current_cds() -> (
    None
):
    # Given
    facts = [
        _soi_cd_fact(
            "adjusted_gross_income",
            10.0,
            geography_id="5001700US0601",
            source_row_id="ca_01",
        ),
        _soi_state_fact(
            "adjusted_gross_income",
            100.0,
            geography_id="0400000US30",
            source_row_id="mt_total",
        ),
    ]
    crosswalk = [
        {
            "source_geography_id": "5001700US0601",
            "target_geography_id": "5001900US0601",
            "weight": 1.0,
        },
        {
            "source_geography_id": "5001700US3000",
            "target_geography_id": "5001900US3001",
            "weight": 2.0,
        },
        {
            "source_geography_id": "5001700US3000",
            "target_geography_id": "5001900US3002",
            "weight": 3.0,
        },
    ]

    # When
    translated = translate_congressional_district_facts_to_current_vintage(
        facts,
        crosswalk,
        crosswalk_basis="block_population",
    )

    # Then
    by_geography = {
        fact["geography"]["id"]: fact
        for fact in translated
        if fact["geography"]["level"] == "congressional_district"
        and str(fact["geography"]["id"]).startswith("5001900US30")
    }
    assert set(by_geography) == {"5001900US3001", "5001900US3002"}
    assert by_geography["5001900US3001"]["value"] == pytest.approx(40.0)
    assert by_geography["5001900US3002"]["value"] == pytest.approx(60.0)
    assert (
        by_geography["5001900US3001"]["lineage"][
            "congressional_district_state_total_proxy_source_geography_id"
        ]
        == "0400000US30"
    )
    assert (
        by_geography["5001900US3001"]["lineage"][
            "congressional_district_vintage_source_geography_id"
        ]
        == "5001700US3000"
    )
    assert by_geography["5001900US3001"]["lineage"][
        "congressional_district_vintage_contributions"
    ][0]["source_record_id"].startswith(
        "irs_soi.ty2023.state_agi.mt_total.adjusted_gross_income"
        ".state_total_proxy_cd.5001700US3000."
    )


def test__given_existing_source_cd_state__then_state_total_proxy_is_not_added() -> None:
    # Given
    facts = [
        _soi_state_fact(
            "adjusted_gross_income",
            100.0,
            geography_id="0400000US30",
            source_row_id="mt_total",
        ),
        _soi_cd_fact(
            "adjusted_gross_income",
            10.0,
            geography_id="5001700US3000",
            source_row_id="mt_00",
        ),
    ]
    crosswalk = [
        {
            "source_geography_id": "5001700US3000",
            "target_geography_id": "5001900US3001",
            "weight": 2.0,
        },
        {
            "source_geography_id": "5001700US3000",
            "target_geography_id": "5001900US3002",
            "weight": 3.0,
        },
    ]

    # When
    translated = translate_congressional_district_facts_to_current_vintage(
        facts,
        crosswalk,
    )

    # Then
    by_geography = {
        fact["geography"]["id"]: fact
        for fact in translated
        if fact["geography"]["level"] == "congressional_district"
    }
    assert set(by_geography) == {"5001900US3001", "5001900US3002"}
    assert by_geography["5001900US3001"]["value"] == pytest.approx(4.0)
    assert by_geography["5001900US3002"]["value"] == pytest.approx(6.0)


def test__given_non_soi_source_cd_fact__then_soi_state_total_proxy_still_applies() -> (
    None
):
    # Given
    facts = [
        _non_soi_cd_fact(
            "population",
            500.0,
            geography_id="5001700US3000",
            source_row_id="mt_00",
        ),
        _soi_cd_fact(
            "adjusted_gross_income",
            10.0,
            geography_id="5001700US0601",
            source_row_id="ca_01",
        ),
        _soi_state_fact(
            "adjusted_gross_income",
            100.0,
            geography_id="0400000US30",
            source_row_id="mt_total",
        ),
    ]
    crosswalk = [
        {
            "source_geography_id": "5001700US0601",
            "target_geography_id": "5001900US0601",
            "weight": 1.0,
        },
        {
            "source_geography_id": "5001700US3000",
            "target_geography_id": "5001900US3001",
            "weight": 2.0,
        },
        {
            "source_geography_id": "5001700US3000",
            "target_geography_id": "5001900US3002",
            "weight": 3.0,
        },
    ]

    # When
    translated = translate_congressional_district_facts_to_current_vintage(
        facts,
        crosswalk,
    )

    # Then
    soi_mt_facts = {
        fact["geography"]["id"]: fact
        for fact in translated
        if fact["geography"]["level"] == "congressional_district"
        and str(fact["geography"]["id"]).startswith("5001900US30")
        and fact["source"]["source_name"] == "irs_soi"
    }
    assert set(soi_mt_facts) == {"5001900US3001", "5001900US3002"}
    assert soi_mt_facts["5001900US3001"]["value"] == pytest.approx(40.0)
    assert soi_mt_facts["5001900US3002"]["value"] == pytest.approx(60.0)


def test__given_state_fact_shape_absent_from_source_cds__then_proxy_is_not_added() -> (
    None
):
    # Given
    unmatched_state_fact = _soi_state_fact(
        "adjusted_gross_income",
        100.0,
        geography_id="0400000US30",
        source_row_id="mt_total",
    )
    unmatched_state_fact["dimensions"] = {
        "income_range": "state_only_range",
        "filing_status": "all",
    }
    facts = [
        _soi_cd_fact(
            "adjusted_gross_income",
            10.0,
            geography_id="5001700US0601",
            source_row_id="ca_01",
        ),
        unmatched_state_fact,
    ]
    crosswalk = [
        {
            "source_geography_id": "5001700US0601",
            "target_geography_id": "5001900US0601",
            "weight": 1.0,
        },
        {
            "source_geography_id": "5001700US3000",
            "target_geography_id": "5001900US3001",
            "weight": 1.0,
        },
    ]

    # When
    translated = translate_congressional_district_facts_to_current_vintage(
        facts,
        crosswalk,
    )

    # Then
    by_geography = {
        fact["geography"]["id"]: fact
        for fact in translated
        if fact["geography"]["level"] == "congressional_district"
    }
    assert set(by_geography) == {"5001900US0601"}


def test__given_state_fact_universe_differs_from_source_cds__then_proxy_is_not_added() -> (
    None
):
    # Given
    itemized_state_fact = _soi_state_fact(
        "adjusted_gross_income",
        100.0,
        geography_id="0400000US30",
        source_row_id="mt_total",
    )
    itemized_state_fact["layout"]["record_set_id"] = (
        "irs_soi.ty2023.itemized_all_returns.mt"
    )
    facts = [
        _soi_cd_fact(
            "adjusted_gross_income",
            10.0,
            geography_id="5001700US0601",
            source_row_id="ca_01",
        ),
        itemized_state_fact,
    ]
    crosswalk = [
        {
            "source_geography_id": "5001700US0601",
            "target_geography_id": "5001900US0601",
            "weight": 1.0,
        },
        {
            "source_geography_id": "5001700US3000",
            "target_geography_id": "5001900US3001",
            "weight": 1.0,
        },
    ]

    # When
    translated = translate_congressional_district_facts_to_current_vintage(
        facts,
        crosswalk,
    )

    # Then
    by_geography = {
        fact["geography"]["id"]: fact
        for fact in translated
        if fact["geography"]["level"] == "congressional_district"
    }
    assert set(by_geography) == {"5001900US0601"}


def test__given_state_proxy_cd_targets__then_compiler_uses_current_cd_surface() -> None:
    # Given
    facts = [
        *_packaged_reference_facts(),
        _soi_cd_fact(
            "adjusted_gross_income",
            10.0,
            geography_id="5001700US0601",
            source_row_id="ca_01",
        ),
        _soi_state_fact(
            "adjusted_gross_income",
            100.0,
            geography_id="0400000US30",
            source_row_id="mt_total",
        ),
    ]
    crosswalk = [
        {
            "source_geography_id": "5001700US0601",
            "target_geography_id": "5001900US0601",
            "weight": 1.0,
        },
        {
            "source_geography_id": "5001700US3000",
            "target_geography_id": "5001900US3001",
            "weight": 2.0,
        },
        {
            "source_geography_id": "5001700US3000",
            "target_geography_id": "5001900US3002",
            "weight": 3.0,
        },
    ]

    # When
    registry = compile_us_fiscal_target_registry(
        facts,
        include_congressional_district_targets=True,
        congressional_district_vintage_crosswalk=crosswalk,
        allow_unaged_dollar_targets=True,
    )

    # Then
    cd_specs = {
        spec.metadata["congressional_district_geoid"]: spec
        for spec in registry.specs
        if spec.metadata.get("source_measure_id") == "adjusted_gross_income"
        and spec.metadata.get("ledger_geography_level") == "congressional_district"
        and str(spec.metadata.get("ledger_geography_id")).startswith("5001900US30")
    }
    assert set(cd_specs) == {"3001", "3002"}
    assert cd_specs["3001"].metadata["ledger_geography_id"] == "5001900US3001"
    assert cd_specs["3001"].value == pytest.approx(40.0)
    assert cd_specs["3002"].value == pytest.approx(60.0)


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


def _non_soi_cd_fact(
    measure_id: str,
    value: float,
    *,
    geography_id: str,
    source_row_id: str,
) -> dict[str, object]:
    fact = _soi_cd_fact(
        measure_id,
        value,
        geography_id=geography_id,
        source_row_id=source_row_id,
    )
    source_record_id = f"census_acs.congressional_district.{source_row_id}.{measure_id}"
    fact["lineage"] = {"source_record_id": source_record_id}
    fact["layout"]["record_set_id"] = "census_acs.congressional_district"
    fact["observed_measure"]["source_name"] = "census_acs"
    fact["observed_measure"]["source_table"] = "ACS congressional district"
    fact["observed_measure"]["source_measure_id"] = measure_id
    fact["source"]["source_name"] = "census_acs"
    fact["source"]["source_table"] = "ACS congressional district"
    return fact


def _soi_state_fact(
    measure_id: str,
    value: float,
    *,
    geography_id: str,
    source_row_id: str,
) -> dict[str, object]:
    source_record_id = f"irs_soi.ty2023.state_agi.{source_row_id}.{measure_id}"
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{source_row_id}.{measure_id}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{source_row_id}.{measure_id}",
        "legacy_fact_key": f"ledger.fact.v1:{source_row_id}.{measure_id}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {
            "level": "state",
            "id": geography_id,
            "vintage": "state_fips",
        },
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "dimensions": {"income_range": "all", "filing_status": "all"},
        "layout": {
            "record_set_id": "irs_soi.ty2023.state_agi",
            "groupby_dimension": "irs_soi.state",
            "groupby_value_id": source_row_id,
            "measure_id": measure_id,
            "source_row_id": source_row_id,
        },
        "observed_measure": {
            "source_name": "irs_soi",
            "source_table": "SOI state AGI",
            "source_measure_id": measure_id,
            "source_concept": measure_id,
            "unit": "usd",
        },
        "source": {
            "source_name": "irs_soi",
            "source_table": "SOI state AGI",
            "source_sha256": "sha",
            "vintage": "tax_year_2023",
            "url": "https://example.org/soi-state",
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
