"""FY2024-25 Ofgem cap pricing and NEED kWh margins from vendored facts (#890 F)."""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.energy_pricing import (
    ENGLAND_AND_WALES_GEOGRAPHY_ID,
    NEED_ENGLAND_WALES_REGION_IDS,
    NEED_PROPERTY_TYPE_IDS_BY_GEOGRAPHY,
    NEED_TENURE_IDS,
    SCOTLAND_GEOGRAPHY_ID,
    cap_rates,
    fiscal_year_quarters,
    income_band_ids,
    kwh_to_spend,
    need_geography,
    need_margins_from_facts,
    pricing_operation,
    spend_to_kwh,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

GB = "K03000001"


def _declared() -> dict:
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    parameters = pricing_operation(stage)
    assert parameters is not None
    return parameters


def test_fiscal_year_quarters() -> None:
    assert fiscal_year_quarters("2024-04-01") == (
        "2024-04-01",
        "2024-07-01",
        "2024-10-01",
        "2025-01-01",
    )
    with pytest.raises(ValueError, match="April"):
        fiscal_year_quarters("2024-01-01")


def test_rates_derive_from_the_published_cap_levels() -> None:
    rates, receipt = cap_rates(_declared())

    def level(
        fuel: str, consumption_level: str, quarter: str, vat: str = "excluding_vat"
    ) -> float:
        rows = [
            row
            for row in vendored_rows(
                "ofgem_price_cap_facts.json",
                concept="ofgem.price_cap.cap_level",
                geography_id=GB,
                dimensions={
                    "fuel": fuel,
                    "consumption_level": consumption_level,
                    "payment_method": "direct_debit",
                    "vat_treatment": vat,
                },
            )
            if row["period_coverage"]["start_date"] == quarter
        ]
        assert len(rows) == 1
        return float(rows[0]["value"])

    # 2024-Q2 GB electricity: (932.22 - 208.91) / 3100 kWh x 1.05 VAT.
    quarter_unit = (
        (
            level("electricity_single_rate", "benchmark_consumption", "2024-04-01")
            - level("electricity_single_rate", "nil_consumption", "2024-04-01")
        )
        / 3100.0
        * 1.05
    )
    assert quarter_unit == pytest.approx(0.2450, abs=1e-4)
    assert receipt["rates_by_region"][GB]["electricity_single_rate"][
        "quarterly_unit_rates"
    ]["2024-04-01"] == pytest.approx(quarter_unit)
    assert receipt["rates_by_region"][GB]["gas"]["quarterly_standing_charges"][
        "2024-04-01"
    ] == pytest.approx(level("gas", "nil_consumption", "2024-04-01") * 1.05)
    # FY2024-25 means: GB electricity 24.05 p/kWh + GBP 220.98, gas 6.02 p + GBP 115.11.
    assert rates.unit_rate[(GB, "electricity_single_rate")] == pytest.approx(
        0.2405, abs=5e-4
    )
    assert rates.standing_charge[(GB, "electricity_single_rate")] == pytest.approx(
        220.98, abs=0.01
    )
    assert rates.unit_rate[(GB, "gas")] == pytest.approx(0.0602, abs=5e-4)
    assert rates.standing_charge[(GB, "gas")] == pytest.approx(115.11, abs=0.01)
    # The published including-VAT levels reproduce the declared rate exactly.
    assert receipt["vat_check"]["min_ratio"] == pytest.approx(1.05, abs=1e-6)
    assert receipt["vat_check"]["max_ratio"] == pytest.approx(1.05, abs=1e-6)
    # Regional spread is material: Yorkshire's standing charge dwarfs London's.
    assert rates.standing_charge[("ofgem:yorkshire", "electricity_single_rate")] > (
        rates.standing_charge[("ofgem:london", "electricity_single_rate")] + 80
    )
    assert rates.region_for("NORTHERN_IRELAND") == GB
    assert rates.region_for("SCOTLAND") == "ofgem:southern_scotland"
    with pytest.raises(KeyError, match="crosswalk"):
        rates.region_for("MARS")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda p: p.update(resource="need_energy_facts.json"),
            "ofgem_price_cap_facts",
        ),
        (lambda p: p.update(crosswalk_resource="other.json"), "crosswalk"),
        (lambda p: p.update(vat_rate=0.2), "does not match the declared"),
        (lambda p: p.pop("vat_rate"), "declares no vat_rate"),
        (lambda p: p.update(fiscal_start="2023-04-01"), "lacks"),
        (lambda p: p.update(payment_method="prepayment"), "lacks"),
    ],
)
def test_declaration_refusals(mutation, match: str) -> None:
    parameters = dict(_declared())
    mutation(parameters)
    with pytest.raises(ValueError, match=match):
        cap_rates(parameters)


def test_spend_and_kwh_round_trip_with_standing_charges() -> None:
    rates, _ = cap_rates(_declared())
    regions = np.array(["LONDON", "YORKSHIRE", "NORTHERN_IRELAND", "SCOTLAND"])
    spend = np.array([1200.0, 100.0, 900.0, 0.0])
    kwh = spend_to_kwh(spend, frs_region=regions, fuel="electricity", rates=rates)
    assert kwh[1] == 0.0  # below the standing charge
    assert kwh[3] == 0.0
    back = kwh_to_spend(kwh, frs_region=regions, fuel="electricity", rates=rates)
    assert back[0] == pytest.approx(1200.0)
    assert back[1] == pytest.approx(
        rates.standing_charge[("ofgem:yorkshire", "electricity_single_rate")]
    )
    gas_spend = np.array([500.0, 0.0, 500.0, 300.0])
    gas_kwh = spend_to_kwh(
        gas_spend, frs_region=regions, fuel="gas", rates=rates, connected=gas_spend > 0
    )
    assert gas_kwh[1] == 0.0
    gas_back = kwh_to_spend(
        gas_kwh, frs_region=regions, fuel="gas", rates=rates, connected=gas_kwh > 0
    )
    assert gas_back[1] == 0.0
    assert gas_back[0] == pytest.approx(500.0)
    assert gas_back[2] == pytest.approx(500.0)


def test_need_margins_come_from_both_geographies_in_kwh() -> None:
    margins = need_margins_from_facts()
    targets = margins.targets
    assert [band for band, _, _ in margins.income_bands] == [
        "less_than_gbp15_000",
        "gbp15_000_gbp19_999",
        "gbp20_000_gbp29_999",
        "gbp30_000_gbp39_999",
        "gbp40_000_gbp49_999",
        "gbp50_000_gbp59_999",
        "gbp60_000_gbp69_999",
        "gbp70_000_gbp99_999",
        "gbp100_000_gbp149_999",
        "gbp150_000_or_more",
    ]
    assert margins.income_bands[0][1:] == (0.0, 15000.0)
    assert margins.income_bands[-1][2] == np.inf
    ew_income = targets["income"][
        (ENGLAND_AND_WALES_GEOGRAPHY_ID, "less_than_gbp15_000")
    ]
    assert ew_income["electricity_kwh"] == pytest.approx(2411.765, abs=1e-2)
    assert ew_income["gas_kwh"] == pytest.approx(7755.080, abs=1e-2)
    assert (SCOTLAND_GEOGRAPHY_ID, "less_than_gbp15_000") in targets["income"]
    assert set(NEED_TENURE_IDS.values()) <= {
        category
        for geography, category in targets["tenure"]
        if geography == ENGLAND_AND_WALES_GEOGRAPHY_ID
    }
    for geography, mapping in NEED_PROPERTY_TYPE_IDS_BY_GEOGRAPHY.items():
        assert set(mapping.values()) <= {
            category
            for row_geography, category in targets["accommodation"]
            if row_geography == geography
        }, geography
    assert (
        NEED_PROPERTY_TYPE_IDS_BY_GEOGRAPHY[ENGLAND_AND_WALES_GEOGRAPHY_ID][
            "HOUSE_TERRACED"
        ]
        == "mid_terrace"
    )
    assert (
        margins.receipt["translations"]["property_type"][SCOTLAND_GEOGRAPHY_ID][
            "HOUSE_TERRACED"
        ]
        == "terraced"
    )
    assert set(NEED_ENGLAND_WALES_REGION_IDS.values()) <= {
        category
        for geography, category in targets["region"]
        if geography == ENGLAND_AND_WALES_GEOGRAPHY_ID
    }
    # Scotland's region margin is its all-dwellings mean; NI has no geography.
    scotland_region = targets["region"][(SCOTLAND_GEOGRAPHY_ID, "all_dwellings")]
    assert (
        scotland_region == targets["income"][(SCOTLAND_GEOGRAPHY_ID, "all_dwellings")]
    )
    assert need_geography(["LONDON", "SCOTLAND", "NORTHERN_IRELAND"]).tolist() == [
        ENGLAND_AND_WALES_GEOGRAPHY_ID,
        SCOTLAND_GEOGRAPHY_ID,
        "",
    ]
    assert income_band_ids([0.0, 14_999.0, 15_000.0, 1e7], margins).tolist() == [
        "less_than_gbp15_000",
        "less_than_gbp15_000",
        "gbp15_000_gbp19_999",
        "gbp150_000_or_more",
    ]
    assert margins.receipt["statistic"] == "mean"
    assert len(margins.receipt["source_record_ids"]) > 100
    with pytest.raises(ValueError, match="NEED margins must come from"):
        need_margins_from_facts("ofgem_price_cap_facts.json")


def test_aggregate_admin_anchors_are_the_band_means_priced_at_the_gb_cap() -> None:
    import json
    from pathlib import Path

    gates = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "packages/microcosm-build/src/microcosm/build/uk/gates.json"
        ).read_text("utf-8")
    )
    anchors = {
        anchor["name"]: anchor
        for gate in gates["gates"]
        if gate["id"] == "uk_aggregate_admin"
        for anchor in gate["parameters"]["anchors"]
    }
    rates, _ = cap_rates(_declared())
    margins = need_margins_from_facts()
    bands = [band for band, _, _ in margins.income_bands]
    for name, fuel, column in (
        (
            "need_electricity_mean_spending",
            "electricity_single_rate",
            "electricity_kwh",
        ),
        ("need_gas_mean_spending", "gas", "gas_kwh"),
    ):
        mean_kwh = float(
            np.mean(
                [
                    margins.targets["income"][(ENGLAND_AND_WALES_GEOGRAPHY_ID, band)][
                        column
                    ]
                    for band in bands
                ]
            )
        )
        expected = (
            mean_kwh * rates.unit_rate[(GB, fuel)] + rates.standing_charge[(GB, fuel)]
        )
        assert anchors[name]["value"] == pytest.approx(expected, abs=1e-4), name
        assert anchors[name]["tolerance"] == pytest.approx(0.15 * expected, abs=1e-4)
        assert anchors[name]["period"] == "2024"
        assert "need_energy_facts.json" in anchors[name]["source"]


@pytest.mark.requires_uk
def test_declared_vat_rate_lockstep_with_the_engine() -> None:
    pytest.importorskip("policyengine_uk")
    from microcosm.build.uk_runtime.donor_uprating import engine_parameter_reader

    parameters = _declared()
    reader = engine_parameter_reader()
    assert (
        reader(parameters["vat_parameter_path"], parameters["vat_period"])
        == (parameters["vat_rate"])
    )
