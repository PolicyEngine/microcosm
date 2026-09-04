"""The incumbent-surface evaluator's joins, statuses and summaries (#762 I9)."""

from __future__ import annotations

import pandas as pd
import pytest

from microcosm.build.uk_runtime.incumbent_surface_evaluation import (
    GSS_REGION_CODES,
    INCUMBENT_LOCAL_METRIC_ALIASES,
    classify_local_rows,
    classify_national_rows,
    evaluation_summary,
    load_incumbent_local_fixture,
    load_incumbent_national_fixture,
    load_uk_data_target_parity,
    match_national_rows,
    national_family_status,
    regional_rollup_estimates,
    render_markdown,
)


def test_fixtures_and_parity_load_from_the_package() -> None:
    national = load_incumbent_national_fixture()
    local = load_incumbent_local_fixture()
    parity = load_uk_data_target_parity()
    assert national["fixture"] == "incumbent_2025" and len(national["rows"]) == 637
    assert local["fixture"] == "incumbent_local_2025" and len(local["rows"]) == 23_545
    assert set(INCUMBENT_LOCAL_METRIC_ALIASES) <= set(local["surface"]["metrics"])
    status = national_family_status(parity)
    assert status["hmrc_spi"][0] == "ported_national"
    assert status["nts"][0] == "routed"


def test_national_rows_match_by_name_then_normalised_name_then_single_cell_contract_id() -> (
    None
):
    fixture = [
        {
            "name": "hmrc/employment_income_income_band_12_570_to_15_000",
            "contract_target_id": "hmrc.spi.employment_income.amount_by_total_income_band",
            "source": "hmrc_spi",
            "family": "hmrc_spi",
            "value": 10.0,
        },
        {
            "name": "dwp/esa_claimants",
            "contract_target_id": "dwp.esa_claimants",
            "source": "dwp",
            "family": "dwp",
            "value": 20.0,
        },
        {
            "name": "dwp/pip_dl_standard_claimants",
            "contract_target_id": "dwp.pip.daily_living_standard_claimants",
            "source": "dwp",
            "family": "dwp",
            "value": 30.0,
        },
        {"name": "nts/vehicles_total", "source": "nts", "family": "nts", "value": 40.0},
    ]
    ours = [
        {
            "name": "hmrc/employment_income_income_band_12_570_to_15_000",
            "contract_target_id": "hmrc.spi.employment_income.amount_by_total_income_band",
            "value": 11.0,
        },
        {
            "name": "dwp.esa_claimants",
            "contract_target_id": "dwp.esa_claimants",
            "value": 21.0,
        },
        {
            "name": "dwp.pip.daily_living_standard",
            "contract_target_id": "dwp.pip.daily_living_standard_claimants",
            "value": 31.0,
        },
    ]
    matched = match_national_rows(fixture, ours)
    assert matched["match"].tolist() == [
        "name",
        "normalised_name",
        "contract_id",
        "unmatched",
    ]
    assert matched["our_target"].tolist()[:3] == [11.0, 21.0, 31.0]
    assert matched["our_name"].isna().tolist() == [False, False, False, True]

    classified = classify_national_rows(
        matched,
        bound_names={"hmrc/employment_income_income_band_12_570_to_15_000"},
        exclusions={
            "dwp.esa_claimants": {
                "tracking": "microcosm#736",
                "expires_on": "2026-10-03",
            }
        },
        source_status={"nts": ("routed", "national_nts_vehicle_counts")},
    )
    assert classified["status"].tolist() == [
        "bound",
        "measure_excluded",
        "compiled_not_bound",
        "not_ported:routed",
    ]
    assert classified["status_detail"].tolist()[1] == "microcosm#736 expires 2026-10-03"
    assert classified["status_detail"].tolist()[3] == "national_nts_vehicle_counts"


def test_local_rows_map_metrics_and_carry_our_cell_status() -> None:
    fixture = [
        {
            "name": "age/20_30@E14000001",
            "metric": "age/20_30",
            "area_type": "constituency",
            "geography_id": "E14000001",
            "period": 2025,
            "value": 100.0,
            "raw_value": 100.0,
            "adjustment_factor": 1.0,
            "boundary_mapped_from_2010": True,
        },
        {
            "name": "voa/council_tax/A@E06000001",
            "metric": "voa/council_tax/A",
            "area_type": "local_authority",
            "geography_id": "E06000001",
            "period": 2025,
            "value": 50.0,
            "raw_value": 50.0,
        },
        {
            "name": "voa/council_tax/H@S12000005",
            "metric": "voa/council_tax/H",
            "area_type": "local_authority",
            "geography_id": "S12000005",
            "period": 2025,
            "value": 5.0,
            "raw_value": 5.0,
        },
        {
            "name": "housing/council_tax_net@E06000001",
            "metric": "housing/council_tax_net",
            "area_type": "local_authority",
            "geography_id": "E06000001",
            "period": 2025,
            "value": 7.0,
            "raw_value": 7.0,
        },
    ]
    membership = {
        "targets": {
            "voa.council_tax_stock.by_area.band_h": {
                "geography_levels": {
                    "local_authority": {
                        "candidates": [
                            {"geography_id": "S12000005", "status": "no_fact_for_area"}
                        ]
                    }
                }
            }
        },
        "signed_deferrals": [
            {
                "target_id": "voa.council_tax_stock.by_area.band_h",
                "geography_level": "local_authority",
                "reason_id": "council_tax_band_h_spine_support_absent",
                "area_ids": ["S12000005"],
            }
        ],
    }
    local = classify_local_rows(
        fixture,
        metric_target_ids={
            "age/20_30": "ons.age.20_30",
            "council_tax/band_a": "voa.council_tax_stock.by_area.band_a",
            "council_tax/band_h": "voa.council_tax_stock.by_area.band_h",
        },
        membership=membership,
        our_metric_names={
            "constituency": ("age/20_30",),
            "la": ("council_tax/band_a", "council_tax/band_h"),
        },
        bound_names={
            "ons.age.20_30@E14000001",
            "voa.council_tax_stock.by_area.band_a@E06000001",
        },
        unmapped_concern={
            "housing/council_tax_net": ("blocked_source", "local_council_tax_net")
        },
    )
    assert local["our_metric"].tolist() == [
        "age/20_30",
        "council_tax/band_a",
        "council_tax/band_h",
        None,
    ]
    assert local["status"].tolist() == [
        "bound",
        "bound",
        "signed_deferred",
        "not_ported:blocked_source",
    ]
    assert (
        local["status_detail"].tolist()[2] == "council_tax_band_h_spine_support_absent"
    )
    assert local["status_detail"].tolist()[3] == "local_council_tax_net"


def test_summary_and_markdown_measure_only_what_has_an_estimate() -> None:
    national = pd.DataFrame(
        [
            {
                "incumbent_name": "a",
                "source": "dwp",
                "status": "bound",
                "incumbent_target": 100.0,
                "candidate_estimate": 105.0,
            },
            {
                "incumbent_name": "b",
                "source": "dwp",
                "status": "measure_excluded",
                "incumbent_target": 100.0,
                "candidate_estimate": 40.0,
            },
            {
                "incumbent_name": "c",
                "source": "nts",
                "status": "not_ported:routed",
                "incumbent_target": 100.0,
                "candidate_estimate": None,
            },
        ]
    )
    local = pd.DataFrame(
        [
            {
                "incumbent_name": "x",
                "area_type": "constituency",
                "incumbent_metric": "age/20_30",
                "status": "bound",
                "incumbent_target": 10.0,
                "candidate_estimate": 10.5,
                "incumbent_estimate": 12.0,
            },
            {
                "incumbent_name": "y",
                "area_type": "local_authority",
                "incumbent_metric": "housing/council_tax_net",
                "status": "not_ported:blocked_source",
                "incumbent_target": 10.0,
                "candidate_estimate": None,
                "incumbent_estimate": None,
            },
        ]
    )
    summary = evaluation_summary(national, local)
    assert summary["national"]["candidate"]["rows"] == 3
    assert summary["national"]["candidate"]["measured"] == 2
    assert summary["national"]["candidate"]["within_10pct"] == 0.5
    assert summary["national"]["candidate"]["by_status"]["measure_excluded"] == 1
    assert summary["local"]["candidate"]["measured"] == 1
    assert summary["local"]["incumbent"]["within_25pct"] == 1.0
    text = render_markdown(summary, national, local)
    assert "The ugly part" in text and "| b | measure_excluded |" in text
    assert "Not measurable on our side: 1 national rows" in text


def test_regional_rollup_measures_age_and_band_rows_by_region() -> None:
    hh = pd.Index([1, 2, 3, 4])
    region = pd.Series(["LONDON", "LONDON", "EAST_OF_ENGLAND", "YORKSHIRE"], index=hh)
    weight = pd.Series([10.0, 20.0, 5.0, 7.0], index=hh)
    band = pd.Series(["A", "H", "A", "D"], index=hh)
    person_hh = pd.Series([1, 1, 2, 3, 4])
    age = pd.Series([5, 40, 9, 89, 90])
    out = regional_rollup_estimates(
        [
            "ons/london_age_0_9",
            "ons/east_age_80_89",
            "ons/yorkshire_and_the_humber_age_80_89",
            "voa/council_tax/LONDON/A",
            "voa/council_tax/LONDON/total",
            "voa/council_tax/EAST_OF_ENGLAND/H",
            "obr/income_tax",
        ],
        household_region=region,
        household_weight=weight,
        household_band=band,
        person_household_id=person_hh,
        person_age=age,
    )
    assert out["ons/london_age_0_9"] == {
        "estimate": 30.0,
        "rollup": "age_band_by_region",
        "region": "LONDON",
    }
    assert out["ons/east_age_80_89"]["estimate"] == 5.0
    assert out["ons/east_age_80_89"]["region"] == "EAST_OF_ENGLAND"
    # The band is closed: age 90 is outside 80_89.
    assert out["ons/yorkshire_and_the_humber_age_80_89"]["estimate"] == 0.0
    assert out["voa/council_tax/LONDON/A"]["estimate"] == 10.0
    assert out["voa/council_tax/LONDON/total"]["estimate"] == 30.0
    assert out["voa/council_tax/LONDON/total"]["rollup"] == "council_tax_band_by_region"
    assert out["voa/council_tax/EAST_OF_ENGLAND/H"]["estimate"] == 0.0
    assert "obr/income_tax" not in out
    with pytest.raises(ValueError, match="not on the frame"):
        regional_rollup_estimates(
            ["voa/council_tax/MARS/A"],
            household_region=region,
            household_weight=weight,
            household_band=band,
            person_household_id=person_hh,
            person_age=age,
        )


def test_gss_region_codes_cover_the_artifact_placeholders() -> None:
    assert GSS_REGION_CODES["E12000007"] == "LONDON"
    assert GSS_REGION_CODES["N99999999"] == "NORTHERN_IRELAND"
    assert GSS_REGION_CODES["W99999999"] == "WALES"
    assert GSS_REGION_CODES["S99999999"] == "SCOTLAND"


def test_local_metric_target_ids_map_metric_name_to_contract_id() -> None:
    from microcosm.build.uk_runtime.ledger_targets import _uk_local_metric_target_ids

    mapping = _uk_local_metric_target_ids()
    # Direction matters: the evaluator looks our metric name up to find the
    # contract id the membership register is keyed by.
    assert mapping["council_tax/band_h"] == "voa.council_tax_stock.by_area.band_h"
    assert mapping["uc_households"] == "dwp.uc.households_by_area"
