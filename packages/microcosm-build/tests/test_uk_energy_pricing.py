"""QEP average prices, published gas connection, NEED 2024 shape and the DESNZ level (#890 F, chronicle#270)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.energy_pricing import (
    ELECTRICITY_KWH,
    ENGLAND_AND_WALES_GEOGRAPHY_ID,
    GAS_KWH,
    NEED_ENGLAND_WALES_REGION_IDS,
    NEED_PROPERTY_TYPE_IDS_BY_GEOGRAPHY,
    NEED_TENURE_IDS,
    SCOTLAND_GEOGRAPHY_ID,
    fiscal_year_quarters,
    impose_gas_connection,
    income_band_ids,
    kwh_to_spend,
    need_geography,
    need_margins_from_facts,
    pricing_operation,
    published_energy_level,
    published_gas_connected_shares,
    qep_prices,
    rake_energy_kwh,
    spend_to_kwh,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

UK = "united_kingdom"
NEED_PERIOD = 2024


def _declared() -> dict:
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    parameters = pricing_operation(stage)
    assert parameters is not None
    return parameters


def _qep_value(concept: str, region: str, component: str, fuel: str) -> float:
    dims = {
        "payment_method": "all",
        "price_component": component,
        "vat_treatment": "including_vat",
        "fuel": fuel,
    }
    if fuel == "electricity":
        dims["metering_arrangement"] = "standard"
    rows = vendored_rows(
        "qep_energy_prices.json",
        concept=concept,
        period_type="fiscal_year",
        period_value=2024,
        groupby_value_id=region,
        dimensions=dims,
    )
    assert len(rows) == 1, (concept, region)
    return float(rows[0]["value"])


def test_fiscal_year_quarters() -> None:
    assert fiscal_year_quarters("2024-04-01") == (
        "2024-04-01",
        "2024-07-01",
        "2024-10-01",
        "2025-01-01",
    )
    with pytest.raises(ValueError, match="April"):
        fiscal_year_quarters("2024-01-01")


def test_prices_are_the_published_qep_averages_paid() -> None:
    prices, receipt = qep_prices(_declared())
    # FY2024-25, all payment methods, including VAT: UK electricity 24.81 p/kWh
    # + GBP 209.39 a year, gas 6.23 p/kWh + GBP 113.13.
    assert prices.unit_rate[(UK, "electricity")] == pytest.approx(0.2481, abs=5e-4)
    assert prices.fixed_cost[(UK, "electricity")] == pytest.approx(209.39, abs=0.01)
    assert prices.unit_rate[(UK, "gas")] == pytest.approx(0.0623, abs=5e-4)
    assert prices.fixed_cost[(UK, "gas")] == pytest.approx(113.13, abs=0.01)
    for region in sorted(set(prices.region_of_frs_region.values()) | {UK}):
        for fuel in ("electricity", "gas"):
            if (region, fuel) not in prices.unit_rate:
                assert (region, fuel) == ("northern_ireland", "gas")
                continue
            assert prices.unit_rate[(region, fuel)] == pytest.approx(
                _qep_value(
                    f"desnz.qep.domestic_{fuel}_variable_unit_cost",
                    region,
                    "variable_unit_cost",
                    fuel,
                )
            )
            if (region, fuel) == ("northern_ireland", "electricity"):
                continue  # no published fixed cost, checked below
            assert prices.fixed_cost[(region, fuel)] == pytest.approx(
                _qep_value(
                    f"desnz.qep.domestic_{fuel}_fixed_cost", region, "fixed_cost", fuel
                )
            )
    # Every FRS region is priced for both fuels; Northern Ireland gas falls to
    # the UK average because QEP publishes no Northern Ireland gas row.
    for frs_region in prices.region_of_frs_region:
        for fuel in ("electricity", "gas"):
            assert (prices.region_for(frs_region, fuel), fuel) in prices.unit_rate
    assert prices.region_for("SCOTLAND", "electricity") == "south_scotland"
    assert prices.region_for("NORTHERN_IRELAND", "electricity") == "northern_ireland"
    assert prices.region_for("NORTHERN_IRELAND", "gas") == UK
    assert prices.unit_rate[("northern_ireland", "electricity")] > (
        prices.unit_rate[(UK, "electricity")] + 0.04
    )
    # QEP publishes no Northern Ireland electricity fixed cost (unit-rate-only
    # standard tariffs): zero, and the receipt says it was not published.
    assert prices.fixed_cost[("northern_ireland", "electricity")] == 0.0
    assert (
        receipt["prices_by_region"]["northern_ireland"]["electricity"][
            "fixed_cost_published"
        ]
        is False
    )
    assert receipt["prices_by_region"][UK]["gas"]["fixed_cost_published"] is True
    assert receipt["gas_priced_at_uk_average"] == ["NORTHERN_IRELAND"]
    assert receipt["payment_method"] == "all"
    assert receipt["vat_treatment"] == "including_vat"
    assert receipt["period_type"] == "fiscal_year" and receipt["period_value"] == 2024
    # Regional spread is material in the fixed cost.
    assert prices.fixed_cost[("north_east", "electricity")] > (
        prices.fixed_cost[("london", "electricity")] + 80
    )
    with pytest.raises(KeyError, match="crosswalk"):
        prices.region_for("MARS", "gas")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda p: p.update(resource="need_energy_facts.json"), "qep_energy_prices"),
        (lambda p: p.update(crosswalk_resource="other.json"), "crosswalk"),
        (lambda p: p.update(vat_treatment="excluding_vat"), "including_vat"),
        (lambda p: p.update(period_type="quarter"), "period_type"),
        (lambda p: p.update(period_value=2019), "expected one vendored row"),
        (lambda p: p.update(payment_method="cheque"), "expected one vendored row"),
    ],
)
def test_price_declaration_refusals(mutation, match: str) -> None:
    parameters = dict(_declared())
    mutation(parameters)
    with pytest.raises(ValueError, match=match):
        qep_prices(parameters)


def test_spend_and_kwh_round_trip_with_fixed_costs() -> None:
    prices, _ = qep_prices(_declared())
    regions = np.array(["LONDON", "NORTH_EAST", "NORTHERN_IRELAND", "SCOTLAND"])
    spend = np.array([1200.0, 100.0, 900.0, 0.0])
    kwh = spend_to_kwh(spend, frs_region=regions, fuel="electricity", prices=prices)
    assert kwh[1] == 0.0  # below the fixed cost
    assert kwh[3] == 0.0
    back = kwh_to_spend(kwh, frs_region=regions, fuel="electricity", prices=prices)
    assert back[0] == pytest.approx(1200.0)
    assert back[1] == pytest.approx(prices.fixed_cost[("north_east", "electricity")])
    assert back[2] == pytest.approx(900.0)
    gas_spend = np.array([500.0, 0.0, 500.0, 300.0])
    gas_kwh = spend_to_kwh(
        gas_spend,
        frs_region=regions,
        fuel="gas",
        prices=prices,
        connected=gas_spend > 0,
    )
    assert gas_kwh[1] == 0.0
    gas_back = kwh_to_spend(
        gas_kwh, frs_region=regions, fuel="gas", prices=prices, connected=gas_kwh > 0
    )
    assert gas_back[1] == 0.0
    assert gas_back[0] == pytest.approx(500.0)
    assert gas_back[2] == pytest.approx(500.0)  # Northern Ireland gas at the UK average
    assert gas_back[3] == pytest.approx(300.0)


def test_qep_averages_paid_sit_near_the_cap_levels() -> None:
    """The cap matrix stays vendored as the regional maximum; QEP is what was paid."""

    prices, _ = qep_prices(_declared())
    quarters = fiscal_year_quarters("2024-04-01")

    def cap(fuel: str, level: str, quarter: str) -> float:
        rows = [
            row
            for row in vendored_rows(
                "ofgem_price_cap_facts.json",
                concept="ofgem.price_cap.cap_level",
                geography_id="K03000001",
                dimensions={
                    "fuel": fuel,
                    "consumption_level": level,
                    "payment_method": "direct_debit",
                    "vat_treatment": "including_vat",
                },
            )
            if row["period_coverage"]["start_date"] == quarter
        ]
        assert len(rows) == 1
        return float(rows[0]["value"])

    benchmark = {"electricity_single_rate": 3100.0, "gas": 12000.0}
    for cap_fuel, fuel in (("electricity_single_rate", "electricity"), ("gas", "gas")):
        unit = np.mean(
            [
                (
                    cap(cap_fuel, "benchmark_consumption", q)
                    - cap(cap_fuel, "nil_consumption", q)
                )
                / benchmark[cap_fuel]
                for q in quarters
            ]
        )
        fixed = np.mean([cap(cap_fuel, "nil_consumption", q) for q in quarters])
        assert abs(prices.unit_rate[(UK, fuel)] / unit - 1.0) < 0.1, fuel
        assert abs(prices.fixed_cost[(UK, fuel)] / fixed - 1.0) < 0.1, fuel


def test_published_gas_connected_shares_are_meters_over_meters() -> None:
    shares, receipt = published_gas_connected_shares(_declared())
    assert set(shares) == set(NEED_ENGLAND_WALES_REGION_IDS) | {
        "SCOTLAND",
        "NORTHERN_IRELAND",
    }
    assert shares["NORTHERN_IRELAND"] is None
    assert receipt["by_region"]["NORTHERN_IRELAND"]["rule"] == "positive_gas_spend"

    def meters(area: str, fuel: str) -> float:
        rows = vendored_rows(
            "desnz_domestic_energy_facts.json",
            concept=f"desnz.subnational.domestic_{fuel}_meter_count",
            geography_id=area,
            period_value=2024,
            dimensions={"metric": "meter_count"},
        )
        assert len(rows) == 1
        return float(rows[0]["value"])

    assert shares["LONDON"] == pytest.approx(
        meters("E12000007", "gas") / meters("E12000007", "electricity")
    )
    assert shares["SCOTLAND"] == pytest.approx(
        meters("S92000003", "gas") / meters("S92000003", "electricity")
    )
    assert shares["WALES"] == pytest.approx(
        meters("W92000004", "gas") / meters("W92000004", "electricity")
    )
    for region, share in shares.items():
        if share is not None:
            assert 0.6 < share < 0.95, region
    assert receipt["by_region"]["LONDON"]["area"] == "E12000007"
    assert len(receipt["by_region"]["LONDON"]["source_record_ids"]) == 2
    for mutation, match in (
        (lambda p: p.update(connection_resource="qep_energy_prices.json"), "desnz"),
        (lambda p: p.update(connection_rule="census"), "connection_rule"),
        (lambda p: p.update(connection_fallback="none"), "connection_fallback"),
        (lambda p: p.update(connection_period_value=2019), "expected one vendored row"),
    ):
        parameters = dict(_declared())
        mutation(parameters)
        with pytest.raises(ValueError, match=match):
            published_gas_connected_shares(parameters)


def test_impose_gas_connection_disconnects_the_smallest_draws_first() -> None:
    gas = np.array([0.0] + list(range(1, 10)) + [5.0] * 10 + [3.0] * 10, dtype=float)
    region = np.array(["LONDON"] * 10 + ["WALES"] * 10 + ["NORTHERN_IRELAND"] * 10)
    weights = np.ones(30)
    connected, receipt = impose_gas_connection(
        gas,
        frs_region=region,
        weights=weights,
        shares={"LONDON": 0.5, "WALES": 1.0, "NORTHERN_IRELAND": None},
    )
    # London: 9 of 10 positive, target 5 of 10: the four smallest draws go.
    assert connected[:10].tolist() == [False] * 5 + [True] * 5
    london = receipt["by_region"]["LONDON"]
    assert london["share_before"] == pytest.approx(0.9)
    assert london["share_after"] == pytest.approx(0.5)
    assert london["rows_disconnected"] == 4 and london["shortfall"] == 0.0
    # Wales sits below its published share: nothing to disconnect, a shortfall.
    assert connected[10:20].all()
    assert receipt["by_region"]["WALES"]["shortfall"] == pytest.approx(0.0)
    assert receipt["by_region"]["WALES"]["rows_disconnected"] == 0
    # Northern Ireland keeps the positive-gas rule.
    assert connected[20:].all()
    assert receipt["by_region"]["NORTHERN_IRELAND"]["rule"] == "positive_gas_spend"
    assert receipt["rows_disconnected"] == 4
    assert receipt["share_before"] == pytest.approx(29 / 30)
    assert receipt["share_after"] == pytest.approx(25 / 30)
    with pytest.raises(ValueError, match="disconnect_rule"):
        impose_gas_connection(
            gas, frs_region=region, weights=weights, shares={}, disconnect_rule="random"
        )


def test_published_level_is_the_fiscal_year_sum_of_energy_trends_quarters() -> None:
    level, receipt = published_energy_level(_declared())
    quarters = fiscal_year_quarters("2024-04-01")
    for column, concept in (
        (ELECTRICITY_KWH, "desnz.energy_trends.domestic_electricity_consumption"),
        (GAS_KWH, "desnz.energy_trends.domestic_gas_consumption"),
    ):
        rows = vendored_rows(
            "desnz_domestic_energy_facts.json",
            concept=concept,
            period_type="quarter",
            dimensions={"temperature_adjustment": "actual_temperature"},
        )
        expected = sum(
            float(r["value"])
            for r in rows
            if r["period_coverage"]["start_date"] in quarters
        )
        assert level[column] == pytest.approx(expected)
        assert receipt["by_fuel"][column]["total_kwh"] == pytest.approx(expected)
    # FY2024-25: about 93 TWh of electricity and 260 TWh of gas.
    assert level[ELECTRICITY_KWH] == pytest.approx(93.03e9, rel=1e-3)
    assert level[GAS_KWH] == pytest.approx(260.07e9, rel=1e-3)
    assert receipt["geography_ids"] == ["K02000001"]
    for mutation, match in (
        (lambda p: p.update(level_resource="need_energy_facts.json"), "level_resource"),
        (lambda p: p.update(level_fiscal_start="2020-04-01"), "lacks quarters"),
        (
            lambda p: p.update(level_temperature_adjustment="weather_corrected"),
            "actual_temperature",
        ),
    ):
        parameters = dict(_declared())
        mutation(parameters)
        with pytest.raises(ValueError, match=match):
            published_energy_level(parameters)


def test_need_margins_come_from_both_geographies_in_kwh() -> None:
    margins = need_margins_from_facts(period_value=NEED_PERIOD)
    targets = margins.targets
    assert margins.period_value == NEED_PERIOD
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
    published = {
        fuel: float(
            vendored_rows(
                "need_energy_facts.json",
                concept=concept,
                geography_id=ENGLAND_AND_WALES_GEOGRAPHY_ID,
                period_value=NEED_PERIOD,
                dimensions={
                    "statistic": "mean",
                    "household_income_band": "less_than_gbp15_000",
                },
            )[0]["value"]
        )
        for fuel, concept in (
            (ELECTRICITY_KWH, "desnz.need.household_electricity_consumption"),
            (GAS_KWH, "desnz.need.household_gas_consumption"),
        )
    }
    assert ew_income == pytest.approx(published)
    # The 2023 consumption year is still vendored and differs from 2024.
    earlier = need_margins_from_facts(period_value=2023)
    assert earlier.period_value == 2023
    assert earlier.targets["income"][
        (ENGLAND_AND_WALES_GEOGRAPHY_ID, "less_than_gbp15_000")
    ] != pytest.approx(ew_income)
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
    assert margins.receipt["period_value"] == NEED_PERIOD
    assert len(margins.receipt["source_record_ids"]) > 100
    with pytest.raises(ValueError, match="NEED margins must come from"):
        need_margins_from_facts("ofgem_price_cap_facts.json", period_value=NEED_PERIOD)
    with pytest.raises(ValueError, match="2019"):
        need_margins_from_facts(period_value=2019)


def _synthetic_rake_receipt(n: int = 1200):
    """A synthetic recipient frame raked like the stage does, with its receipt."""

    declared = _declared()
    margins = need_margins_from_facts(period_value=NEED_PERIOD)
    level, _ = published_energy_level(declared)
    shares, _ = published_gas_connected_shares(declared)
    rng = np.random.default_rng(5)
    table = pd.DataFrame(
        {
            ELECTRICITY_KWH: rng.uniform(1000.0, 5000.0, n),
            GAS_KWH: np.where(
                np.arange(n) % 20 == 0, 0.0, rng.uniform(5000.0, 20000.0, n)
            ),
        }
    )
    region = np.array(["LONDON", "WALES", "SCOTLAND"], dtype=object)[np.arange(n) % 3]
    income = rng.uniform(5e3, 2e5, n)
    tenure = np.array(["OWNED_OUTRIGHT", "RENT_PRIVATELY"], dtype=object)[
        np.arange(n) % 2
    ]
    accommodation = np.array(["HOUSE_DETACHED", "FLAT"], dtype=object)[
        (np.arange(n) // 2) % 2
    ]
    weights = rng.uniform(0.5, 2.0, n)
    connected, connection = impose_gas_connection(
        table[GAS_KWH].to_numpy(), frs_region=region, weights=weights, shares=shares
    )
    table.loc[~connected, GAS_KWH] = 0.0
    raked, receipt = rake_energy_kwh(
        table,
        margins=margins,
        frs_region=region,
        income=income,
        weights=weights,
        iterations=50,
        tenure=tenure,
        accommodation=accommodation,
        use_region_margin=True,
        gas_connected=connected,
        level=level,
    )
    receipt["gas_connection"] = {"rule": "published_meter_share", **connection}
    return margins, level, raked, weights, receipt


def test_rake_fit_targets_lockstep_with_the_vendored_need_rows() -> None:
    """Every shape target is the vendored NEED 2024 mean; the target is shape x level."""

    margins, level, raked, weights, receipt = _synthetic_rake_receipt()
    fit = receipt["fit"]
    assert set(fit) == {"income", "tenure", "accommodation", "region"}
    assert receipt["margins_period_value"] == NEED_PERIOD
    checked = 0
    for margin, block in fit.items():
        for key, cell in block["cells"].items():
            geography, category = key.split(":", 1)
            rows = {
                fuel: float(
                    vendored_rows(
                        "need_energy_facts.json",
                        concept=concept,
                        geography_id=geography,
                        period_value=NEED_PERIOD,
                        dimensions={"statistic": "mean", dimension: category},
                    )[0]["value"]
                )
                for fuel, concept in (
                    (ELECTRICITY_KWH, "desnz.need.household_electricity_consumption"),
                    (GAS_KWH, "desnz.need.household_gas_consumption"),
                )
                for dimension in (
                    {
                        "income": "household_income_band",
                        "tenure": "tenure",
                        "accommodation": "property_type",
                        "region": "region"
                        if geography == "K04000001"
                        else "household_income_band",
                    }[margin],
                )
            }
            for fuel in (ELECTRICITY_KWH, GAS_KWH):
                assert cell[fuel]["shape_target"] == pytest.approx(rows[fuel]), (
                    margin,
                    key,
                )
                assert cell[fuel]["level_factor"] == receipt["level_factor"][fuel]
                assert cell[fuel]["target"] == pytest.approx(
                    rows[fuel] * receipt["level_factor"][fuel]
                )
            checked += 1
    assert checked >= 10
    # Region is swept last, so it fits exactly; gas is averaged over its
    # connected rows, a smaller weight than electricity's.
    assert fit["region"]["max_abs_relative_deviation"][ELECTRICITY_KWH] < 1e-9
    assert fit["region"]["max_abs_relative_deviation"][GAS_KWH] < 1e-9
    london = fit["region"]["cells"][f"{ENGLAND_AND_WALES_GEOGRAPHY_ID}:london"]
    assert london[GAS_KWH]["weighted_rows"] < london[ELECTRICITY_KWH]["weighted_rows"]


def test_rake_levels_the_frame_to_the_published_totals() -> None:
    _, level, raked, weights, receipt = _synthetic_rake_receipt()
    for fuel in (ELECTRICITY_KWH, GAS_KWH):
        total = float(np.dot(raked[fuel].to_numpy(dtype=float), weights))
        assert total == pytest.approx(level[fuel], rel=1e-9)
        block = receipt["level"][fuel]
        assert block["published_kwh"] == level[fuel]
        assert block["factor"] == pytest.approx(
            level[fuel] / block["frame_kwh_before"], rel=1e-12
        )
        assert block["frame_kwh_after"] == pytest.approx(level[fuel], rel=1e-9)
        assert receipt["level_factor"][fuel] == block["factor"]
    # Without a level the factors are one and the shape targets are the targets.
    margins = need_margins_from_facts(period_value=NEED_PERIOD)
    table = pd.DataFrame({ELECTRICITY_KWH: [1.0, 2.0], GAS_KWH: [3.0, 4.0]})
    _, plain = rake_energy_kwh(
        table,
        margins=margins,
        frs_region=np.array(["LONDON", "LONDON"]),
        income=np.array([1e4, 1e4]),
        weights=None,
        iterations=1,
    )
    assert plain["level"] is None and plain["level_factor"] == {
        ELECTRICITY_KWH: 1.0,
        GAS_KWH: 1.0,
    }


def test_energy_is_checked_at_stage_time_by_the_energy_rake_gate() -> None:
    """NEED shape, DESNZ level and the connection share are stage-health facts (#890)."""

    from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

    gates = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "packages/microcosm-build/src/microcosm/build/uk/gates.json"
        ).read_text("utf-8")
    )
    by_id = {gate["id"]: gate for gate in gates["gates"]}
    assert [
        a["name"] for a in by_id["uk_aggregate_admin"]["parameters"]["anchors"]
    ] == ["nhs_spending_total"]
    parameters = by_id["uk_stage_lcfs_consumption_energy_rake"]["parameters"]
    assert parameters["check"] == "energy_rake"
    assert parameters["margins_period_value"] == NEED_PERIOD
    assert by_id["uk_stage_lcfs_consumption_energy_rake"]["evidence_absent_blocks"]
    _, _, _, _, receipt = _synthetic_rake_receipt()
    evidence = {"stage": "lcfs_consumption", "energy_rake": receipt}

    def run(params=parameters, ev=evidence):
        return uk_stage_health_gate(
            evidence=ev,
            stage="lcfs_consumption",
            check="energy_rake",
            parameters=params,
        )

    def failing(params=parameters, ev=evidence, *, match: str):
        result = run(params, ev)
        assert not result.passed and any(match in f for f in result.failures), (
            match,
            result.failures,
        )

    generous = {**parameters, "maximum_relative_deviation": 1.0}
    passed = run(generous)
    assert passed.passed, passed.failures
    assert passed.details["worst"]["region:gas_kwh"] < 1e-9
    assert passed.details["cells_fact_checked"] > 0
    assert set(passed.details["level_factor"]) == {ELECTRICITY_KWH, GAS_KWH}
    assert set(passed.details["connected_share"]) == {"LONDON", "WALES", "SCOTLAND"}
    failing(
        {**parameters, "maximum_relative_deviation": 0.0}, match="residual tolerance"
    )
    failing(
        {**generous, "margins": ["income", "tenure", "accommodation"]},
        match="undeclared margins",
    )
    trimmed = copy.deepcopy(receipt)
    trimmed["margins"] = ["income", "tenure", "accommodation"]
    del trimmed["fit"]["region"]
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": trimmed},
        match="was not raked",
    )
    tampered = copy.deepcopy(receipt)
    key = next(iter(tampered["fit"]["income"]["cells"]))
    tampered["fit"]["income"]["cells"][key][GAS_KWH]["shape_target"] *= 1.01
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": tampered},
        match="not the vendored NEED mean",
    )
    tampered = copy.deepcopy(receipt)
    tampered["fit"]["income"]["cells"][key][GAS_KWH]["target"] *= 1.01
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": tampered},
        match="times the level factor",
    )
    tampered = copy.deepcopy(receipt)
    tampered["level"][ELECTRICITY_KWH]["published_kwh"] *= 1.01
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": tampered},
        match="not the vendored DESNZ total",
    )
    tampered = copy.deepcopy(receipt)
    tampered["level"][GAS_KWH]["frame_kwh_after"] *= 1.01
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": tampered},
        match="frame total after levelling",
    )
    tampered = copy.deepcopy(receipt)
    tampered["level_factor"][GAS_KWH] *= 1.01
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": tampered},
        match="not the published total over the frame total",
    )
    failing(
        generous,
        {
            "stage": "lcfs_consumption",
            "energy_rake": {**receipt, "margins_period_value": 2023},
        },
        match="not the declared consumption year",
    )
    zeroed = {
        **receipt,
        "zero_current_cells": [{"margin": "_need_income", "category": "x"}],
    }
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": zeroed},
        match="zero current mean",
    )
    failing(
        generous,
        {
            "stage": "lcfs_consumption",
            "energy_rake": {**receipt, "gas_rake_population": "all_rows"},
        },
        match="gas_connected_rows",
    )
    failing(
        generous,
        {
            "stage": "lcfs_consumption",
            "energy_rake": {k: v for k, v in receipt.items() if k != "fit"},
        },
        match="no fit block",
    )
    failing(
        generous,
        {
            "stage": "lcfs_consumption",
            "energy_rake": {k: v for k, v in receipt.items() if k != "level"},
        },
        match="no level block",
    )
    failing(
        generous,
        {
            "stage": "lcfs_consumption",
            "energy_rake": {k: v for k, v in receipt.items() if k != "gas_connection"},
        },
        match="not imposed at the published meter share",
    )
    tampered = copy.deepcopy(receipt)
    region = next(
        r
        for r, entry in tampered["gas_connection"]["by_region"].items()
        if entry.get("shortfall") == 0.0 and entry.get("rows_disconnected", 0) > 0
    )
    tampered["gas_connection"]["by_region"][region]["share_after"] += 0.1
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": tampered},
        match="not the published",
    )
    with pytest.raises(ValueError, match="declares no margins"):
        run({**generous, "margins": []})
    with pytest.raises(ValueError, match="differs from the stage"):
        run({**generous, "margins_period_value": 2023})


def test_gas_is_raked_over_connected_rows_and_electricity_over_all() -> None:
    margins = need_margins_from_facts(period_value=NEED_PERIOD)
    n = 200
    rng = np.random.default_rng(8)
    table = pd.DataFrame(
        {
            ELECTRICITY_KWH: rng.uniform(1000.0, 5000.0, n),
            GAS_KWH: np.where(
                np.arange(n) % 4 == 0, 0.0, rng.uniform(5000.0, 20000.0, n)
            ),
        }
    )
    region = np.full(n, "LONDON", dtype=object)
    income = np.full(n, 10_000.0)  # one E&W income band: less_than_gbp15_000
    connected = table[GAS_KWH].to_numpy() > 0
    raked, receipt = rake_energy_kwh(
        table,
        margins=margins,
        frs_region=region,
        income=income,
        weights=None,
        iterations=1,
        gas_connected=connected,
    )
    cell = margins.targets["income"][
        (ENGLAND_AND_WALES_GEOGRAPHY_ID, "less_than_gbp15_000")
    ]
    assert raked[ELECTRICITY_KWH].mean() == pytest.approx(cell[ELECTRICITY_KWH])
    # Gas: the connected rows carry the NEED mean; unconnected rows stay zero.
    assert raked.loc[connected, GAS_KWH].mean() == pytest.approx(cell[GAS_KWH])
    assert (raked.loc[~connected, GAS_KWH] == 0.0).all()
    assert raked[GAS_KWH].mean() == pytest.approx(cell[GAS_KWH] * connected.mean())
    assert receipt["gas_rake_population"] == "gas_connected_rows"
    assert receipt["gas_connected_rows"] == int(connected.sum())
    # Without a mask every row counts as connected: the all-row mean fits.
    unmasked, _ = rake_energy_kwh(
        table,
        margins=margins,
        frs_region=region,
        income=income,
        weights=None,
        iterations=1,
    )
    assert unmasked[GAS_KWH].mean() == pytest.approx(cell[GAS_KWH])
