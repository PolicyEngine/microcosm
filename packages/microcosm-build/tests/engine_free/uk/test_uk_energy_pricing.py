"""QEP average prices, published gas connection, NEED 2024 shape and the DESNZ level (#890 F, chronicle#270)."""

from __future__ import annotations

import copy
import json

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
    cap_household_kwh,
    declared_household_kwh_ceilings,
    declared_zero_kwh_cells_at_need_mean,
    fiscal_year_quarters,
    impose_gas_connection,
    income_band_ids,
    kwh_to_spend,
    need_geography,
    need_margins_from_facts,
    period_quarters,
    pricing_operation,
    published_energy_level,
    published_gas_connected_shares,
    qep_prices,
    rake_energy_kwh,
    spend_to_kwh,
)
from microcosm.build.uk_runtime.lcfs_consumption import (
    _recipient_energy_rake_iterations,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

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
        period_type="calendar_year",
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


def test_period_quarters_cover_a_calendar_or_fiscal_year() -> None:
    assert period_quarters("calendar_year", 2024) == (
        "2024-01-01",
        "2024-04-01",
        "2024-07-01",
        "2024-10-01",
    )
    assert period_quarters("fiscal_year", 2024) == fiscal_year_quarters("2024-04-01")
    with pytest.raises(ValueError, match="calendar_year or fiscal_year"):
        period_quarters("quarter", 2024)


def test_prices_are_the_published_qep_averages_paid() -> None:
    prices, receipt = qep_prices(_declared())
    # Calendar 2024, the base year of every LCFS spend column (microcosm#1113),
    # all payment methods, including VAT: UK electricity 25.83 p/kWh + GBP
    # 205.11 a year, gas 6.54 p/kWh + GBP 112.95.
    assert prices.unit_rate[(UK, "electricity")] == pytest.approx(0.2583, abs=5e-4)
    assert prices.fixed_cost[(UK, "electricity")] == pytest.approx(205.11, abs=0.01)
    assert prices.unit_rate[(UK, "gas")] == pytest.approx(0.0654, abs=5e-4)
    assert prices.fixed_cost[(UK, "gas")] == pytest.approx(112.95, abs=0.01)
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
    assert receipt["period_type"] == "calendar_year" and receipt["period_value"] == 2024
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
    # The cap matrix is vendored for FY2024-25 only; the calendar-2024 averages
    # paid sit within the same 10% of it.
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
    # The fallback is declared, not inferred: the stage names the regions the
    # crosswalk maps to no subnational area, and the receipt echoes them.
    assert receipt["fallback_regions"] == ["NORTHERN_IRELAND"]

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
        # A declared fallback list that names a region the crosswalk maps to an
        # area, omits one it maps to none, or is absent altogether is refused.
        (
            lambda p: p.update(connection_fallback_regions=["WALES"]),
            "connection_fallback_regions must be exactly",
        ),
        (
            lambda p: p.update(
                connection_fallback_regions=["NORTHERN_IRELAND", "WALES"]
            ),
            "connection_fallback_regions must be exactly",
        ),
        (
            lambda p: p.pop("connection_fallback_regions"),
            "connection_fallback_regions must name",
        ),
        (
            lambda p: p.update(connection_fallback_regions=[]),
            "connection_fallback_regions must name",
        ),
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


def test_impose_gas_connection_skips_a_household_whose_weight_would_overshoot() -> None:
    """The greedy walk with unequal weights: skip, then take the nearest skipped.

    One region, every household gas-positive, weights 1, 1, 2.5, 1, 1 (total
    6.5) and a published share of 0.2, so 5.2 of weight must go. Walking up
    the drawn gas: the two lightest households go (remaining 3.2), the 2.5
    household fits and goes (0.7), the last two would overshoot and are
    skipped; after the walk the skipped household nearest the remainder
    (weight 1, |1 - 0.7| = 0.3 < 0.7) is taken because it brings the share
    nearer the published one, and the other stays connected and is counted
    as skipped for weight.
    """

    gas = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([1.0, 1.0, 2.5, 1.0, 1.0])
    region = np.array(["LONDON"] * 5)
    connected, receipt = impose_gas_connection(
        gas, frs_region=region, weights=weights, shares={"LONDON": 0.2}
    )
    assert connected.tolist() == [False, False, False, False, True]
    london = receipt["by_region"]["LONDON"]
    assert london["share_before"] == pytest.approx(1.0)
    assert london["share_after"] == pytest.approx(1.0 / 6.5)
    assert london["rows_disconnected"] == 4
    assert london["rows_skipped_for_weight"] == 1
    assert london["weight_disconnected"] == pytest.approx(5.5)
    assert london["shortfall"] == 0.0
    # The same walk where taking the nearest skipped household would move the
    # share further from the published one leaves it connected: weights
    # 1, 1, 10, 1, 1 (total 14) and a share of 0.5 remove 7; the 10 household
    # is skipped and, at |10 - 3| = 7, not nearer than the remainder of 3.
    weights = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
    connected, receipt = impose_gas_connection(
        gas, frs_region=region, weights=weights, shares={"LONDON": 0.5}
    )
    assert connected.tolist() == [False, False, True, False, False]
    london = receipt["by_region"]["LONDON"]
    assert london["share_after"] == pytest.approx(10.0 / 14.0)
    assert london["rows_disconnected"] == 4
    assert london["rows_skipped_for_weight"] == 1


def test_identity_uniform_order_spreads_disconnection_across_draw_sizes() -> None:
    """Every group loses the same expected share of its connected mass.

    One region, half the households drawing little gas (a flat-like group)
    and half drawing a lot, all connected, published share 0.6. Walking the
    smallest draws first takes the whole 0.4 from the low-draw group; the
    identity-keyed uniform order takes about 0.4 of each group.
    """

    n = 4000
    gas = np.where(np.arange(n) < n // 2, 2000.0, 15000.0) + np.arange(n)
    region = np.array(["SCOTLAND"] * n)
    weights = np.ones(n)
    ids = np.arange(1, n + 1) * 100
    low = np.arange(n) < n // 2
    lowest_first, _ = impose_gas_connection(
        gas, frs_region=region, weights=weights, shares={"SCOTLAND": 0.6}
    )
    assert not lowest_first[low][: int(0.4 * n)].any()
    assert lowest_first[~low].all()
    connected, receipt = impose_gas_connection(
        gas,
        frs_region=region,
        weights=weights,
        shares={"SCOTLAND": 0.6},
        disconnect_rule="identity_uniform_order",
        identity=ids,
        seed=0,
    )
    assert connected.mean() == pytest.approx(0.6)
    for group in (low, ~low):
        assert connected[group].mean() == pytest.approx(0.6, abs=0.03)
    assert receipt["disconnect_rule"] == "identity_uniform_order"
    assert receipt["seed"] == 0
    assert receipt["salt"] == "lcfs_consumption:gas_disconnection"


def test_identity_uniform_order_is_keyed_by_identity_not_row_order() -> None:
    rng = np.random.default_rng(3)
    n = 600
    gas = rng.uniform(100.0, 20000.0, n)
    region = rng.choice(["LONDON", "WALES", "SCOTLAND"], n)
    weights = rng.uniform(0.5, 3.0, n)
    ids = rng.permutation(np.arange(10_000, 10_000 + n))
    shares = {"LONDON": 0.7, "WALES": 0.8, "SCOTLAND": 0.75}
    kwargs = {"shares": shares, "disconnect_rule": "identity_uniform_order", "seed": 0}
    first, _ = impose_gas_connection(
        gas, frs_region=region, weights=weights, identity=ids, **kwargs
    )
    order = rng.permutation(n)
    second, _ = impose_gas_connection(
        gas[order],
        frs_region=region[order],
        weights=weights[order],
        identity=ids[order],
        **kwargs,
    )
    assert dict(zip(ids, first, strict=True)) == dict(
        zip(ids[order], second, strict=True)
    )
    reseeded, _ = impose_gas_connection(
        gas,
        frs_region=region,
        weights=weights,
        identity=ids,
        **{**kwargs, "seed": 1},
    )
    assert not np.array_equal(first, reseeded)
    with pytest.raises(ValueError, match="needs one identity per row"):
        impose_gas_connection(gas, frs_region=region, weights=weights, **kwargs)


def test_connect_walk_fills_a_region_below_its_share_by_gas_payer_probability() -> None:
    # microcosm#1164: a region whose draw falls short of its published share
    # connects zero-gas households, drawn in proportion to their probability of
    # paying for gas; a household of probability zero is never connected.
    rng = np.random.default_rng(11)
    n = 4000
    region = np.array(["LONDON"] * n, dtype=object)
    weights = rng.uniform(0.5, 1.5, n)
    gas = np.where(np.arange(n) % 10 < 7, 5000.0, 0.0)  # 70 % drawn connected
    likely = np.arange(n) % 10 == 7
    never = np.arange(n) % 10 == 9
    probability = np.where(likely, 0.9, np.where(never, 0.0, 0.1))
    ids = np.arange(n) + 100
    connected, receipt = impose_gas_connection(
        gas,
        frs_region=region,
        weights=weights,
        shares={"LONDON": 0.8},
        disconnect_rule="identity_uniform_order",
        identity=ids,
        connect_probability=probability,
        connect_salt="lcfs_gas_connection",
    )
    entry = receipt["by_region"]["LONDON"]
    assert entry["share_after"] == pytest.approx(0.8, abs=0.001)
    assert entry["shortfall"] == 0.0 and entry["candidates_exhausted"] is False
    assert entry["rows_connected"] == receipt["rows_connected"] > 0
    assert connected[gas > 0].all()
    assert not connected[never].any()
    added = connected & ~(gas > 0)
    # The likely payers take most of the new connections.
    assert added[likely].sum() > 3 * added[~likely].sum()
    assert receipt["connect_rule"] == "gas_payer_probability_order"
    assert receipt["connect_salt"] == "lcfs_gas_connection"
    # Too few candidates: everyone with a probability is connected and the
    # remaining shortfall is receipted as exhausted.
    short, exhausted = impose_gas_connection(
        gas,
        frs_region=region,
        weights=weights,
        shares={"LONDON": 0.95},
        disconnect_rule="identity_uniform_order",
        identity=ids,
        connect_probability=probability,
        connect_salt="lcfs_gas_connection",
    )
    tail = exhausted["by_region"]["LONDON"]
    assert tail["candidates_exhausted"] is True and tail["shortfall"] > 0
    assert short[~never].all() and not short[never].any()
    with pytest.raises(ValueError, match="connect_salt"):
        impose_gas_connection(
            gas,
            frs_region=region,
            weights=weights,
            shares={"LONDON": 0.8},
            identity=ids,
            connect_probability=probability,
        )


def test_the_energy_rake_gate_holds_a_connect_rule_to_its_shortfalls() -> None:
    # A region left short with households still to connect fails; one that
    # ran out of candidates may stand short (microcosm#1164).
    from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

    gates = json.loads(
        (
            _TEST_PATHS.repository
            / "packages/microcosm-build/src/microcosm/build/uk/gates.json"
        ).read_text("utf-8")
    )
    parameters = {gate["id"]: gate["parameters"] for gate in gates["gates"]}[
        "uk_stage_lcfs_consumption_energy_rake"
    ]
    generous = {**parameters, "maximum_relative_deviation": 1.0}
    _, _, _, _, receipt = _synthetic_rake_receipt()

    def run(ev):
        return uk_stage_health_gate(
            evidence={"stage": "lcfs_consumption", "energy_rake": ev},
            stage="lcfs_consumption",
            check="energy_rake",
            parameters=generous,
        )

    assert run(receipt).passed
    short = copy.deepcopy(receipt)
    short["gas_connection"]["by_region"]["LONDON"].update(
        shortfall=0.02, candidates_exhausted=False
    )
    result = run(short)
    assert not result.passed
    assert any("households left to connect" in f for f in result.failures)
    short["gas_connection"]["by_region"]["LONDON"]["candidates_exhausted"] = True
    assert run(short).passed
    renamed = copy.deepcopy(receipt)
    renamed["gas_connection"]["connect_salt"] = "other"
    result = run(renamed)
    assert not result.passed
    assert any("connect rule" in f for f in result.failures)


def _energy_trends_only() -> dict:
    """The declared pricing with no fuel levelled to ONS household spending."""

    parameters = dict(_declared())
    parameters.pop("spend_level_fuels")
    parameters.pop("spend_level_resource")
    return parameters


def test_published_level_is_the_calendar_year_sum_of_energy_trends_quarters() -> None:
    level, receipt = published_energy_level(_energy_trends_only())
    quarters = period_quarters("calendar_year", 2024)
    assert receipt["quarters"] == list(quarters)
    assert (receipt["period_type"], receipt["period_value"]) == ("calendar_year", 2024)
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
    # Calendar 2024: about 91.7 TWh of electricity and 252.3 TWh of gas.
    assert level[ELECTRICITY_KWH] == pytest.approx(91.74e9, rel=1e-3)
    assert level[GAS_KWH] == pytest.approx(252.34e9, rel=1e-3)
    assert receipt["geography_ids"] == ["K02000001"]
    for mutation, match in (
        (lambda p: p.update(level_resource="need_energy_facts.json"), "level_resource"),
        (lambda p: p.update(level_period_value=2019), "lacks quarters"),
        (
            lambda p: p.update(level_period_type="quarter"),
            "calendar_year or fiscal_year",
        ),
        (
            lambda p: p.update(level_temperature_adjustment="weather_corrected"),
            "actual_temperature",
        ),
    ):
        parameters = dict(_declared())
        mutation(parameters)
        with pytest.raises(ValueError, match=match):
            published_energy_level(parameters)


def _one_vendored(resource: str, **criteria) -> dict:
    rows = vendored_rows(resource, **criteria)
    assert len(rows) == 1, criteria
    return rows[0]


def test_the_electricity_level_is_the_kwh_ons_household_spending_buys() -> None:
    """Electricity's level is ONS 04.5.1 less every meter's standing charge, over
    the unit cost, at QEP's UK average prices (microcosm#1418); gas keeps Energy
    Trends."""

    declared = _declared()
    assert declared["spend_level_fuels"] == ["electricity"]
    level, receipt = published_energy_level(declared)
    trends, _ = published_energy_level(_energy_trends_only())
    spend = _one_vendored(
        "ons_household_expenditure_facts.json",
        concept="ons.household_expenditure.electricity",
        geography_id="K02000001",
        period_type="calendar_year",
        period_value=2024,
        dimensions={"coicop": "04.5.1", "consumption_concept": "domestic"},
    )
    meters = sum(
        float(
            _one_vendored(
                "desnz_domestic_energy_facts.json",
                concept="desnz.subnational.domestic_electricity_meter_count",
                geography_id=geography,
                period_type=period_type,
                period_value=2024,
                dimensions={"domestic_scope": "all_domestic"},
            )["value"]
        )
        for geography, period_type in (
            ("K03000001", "calendar_year"),
            ("N92000002", "fiscal_year"),
        )
    )
    unit = _qep_value(
        "desnz.qep.domestic_electricity_variable_unit_cost",
        UK,
        "variable_unit_cost",
        "electricity",
    )
    fixed = _qep_value(
        "desnz.qep.domestic_electricity_fixed_cost", UK, "fixed_cost", "electricity"
    )
    expected = (float(spend["value"]) - meters * fixed) / unit
    assert level[ELECTRICITY_KWH] == pytest.approx(expected, rel=1e-12)
    # Calendar 2024: ONS's GBP 28.13bn buys about 84.9 TWh, 7.4% below Energy
    # Trends' 91.7 TWh; about 30.2 million meters pay the standing charges.
    assert level[ELECTRICITY_KWH] == pytest.approx(84.94e9, rel=1e-3)
    assert meters == pytest.approx(30.19e6, rel=1e-3)
    # Gas keeps Energy Trends: ONS 04.5.2 counts the bottled gas the frame
    # holds as liquid fuel (María's ruling of 2026-10-29).
    assert level[GAS_KWH] == trends[GAS_KWH]
    block = receipt["by_fuel"][ELECTRICITY_KWH]
    assert block["basis"] == "ons_household_spend"
    assert block["coicop"] == "04.5.1"
    assert block["spend_gbp"] == float(spend["value"])
    assert block["energy_trends_kwh"] == trends[ELECTRICITY_KWH]
    assert block["total_kwh"] == level[ELECTRICITY_KWH]
    assert block["domestic_meters"]["meters"] == meters
    # Northern Ireland's electricity meters stay in the subtraction, and the
    # block says why (Anthony's review of microcosm#1420).
    assert "northern_ireland_meters" in block
    assert receipt["by_fuel"][GAS_KWH]["basis"] == "energy_trends_volume"
    assert receipt["spend_level_fuels"] == ["electricity"]
    both = {**declared, "spend_level_fuels": ["electricity", "gas"]}
    gas_level, gas_receipt = published_energy_level(both)
    assert gas_receipt["by_fuel"][GAS_KWH]["coicop"] == "04.5.2"
    assert "northern_ireland_meters" not in gas_receipt["by_fuel"][GAS_KWH]
    assert gas_level[GAS_KWH] == pytest.approx(236.47e9, rel=1e-3)
    for mutation, match in (
        (lambda p: p.update(spend_level_fuels=["electricity", "oil"]), "at most once"),
        (
            lambda p: p.update(spend_level_fuels=["electricity", "electricity"]),
            "at most once",
        ),
        (lambda p: p.update(spend_level_fuels=[]), "non-empty list"),
        (
            lambda p: p.update(spend_level_resource="excise_facts.json"),
            "spend_level_resource",
        ),
        (lambda p: p.update(level_period_type="fiscal_year"), "calendar years"),
        (lambda p: p.update(period_value=2025), "level's year"),
    ):
        parameters = dict(declared)
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


def _declared_sweeps() -> int:
    """The recipient rake's sweep count as the packaged stage declares it."""

    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    return _recipient_energy_rake_iterations(stage)


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
    # Weights summing to about the UK's households, so the DESNZ level gives
    # each household a plausible kWh under the declared domestic ceilings.
    weights = rng.uniform(0.5, 2.0, n)
    weights *= 28_500_000.0 / weights.sum()
    drawn = table[GAS_KWH].to_numpy()
    connected, connection = impose_gas_connection(
        drawn,
        frs_region=region,
        weights=weights,
        shares=shares,
        identity=np.arange(n),
        connect_probability=np.full(n, 0.5),
        connect_salt=declared["connect_salt"],
    )
    table.loc[~connected, GAS_KWH] = 0.0
    table.loc[connected & ~(drawn > 0), GAS_KWH] = 8000.0
    raked, receipt = rake_energy_kwh(
        table,
        margins=margins,
        frs_region=region,
        income=income,
        weights=weights,
        iterations=_declared_sweeps(),
        tenure=tenure,
        accommodation=accommodation,
        use_region_margin=True,
        gas_connected=connected,
        level=level,
        ceilings=declared_household_kwh_ceilings(declared),
    )
    receipt["gas_connection"] = {"rule": "published_meter_share", **connection}
    return margins, level, raked, weights, receipt


def test_rake_receipt_records_one_cross_margin_residual_per_sweep() -> None:
    """The IPF's terminal residual is shown to be converged, not truncated.

    ``iterative_proportional_fit`` runs the stage's declared sweeps (200 since
    microcosm#1012); the receipt carries the maximum absolute relative cell
    deviation after every sweep for each fuel, and the last value is the
    residual the ``energy_rake`` gate holds to its declared tolerance, once
    the gate has checked the series is flat over its convergence window.
    """

    _, _, _, _, receipt = _synthetic_rake_receipt()
    residuals = receipt["sweep_residuals"]
    assert set(residuals) == {ELECTRICITY_KWH, GAS_KWH}
    assert _declared_sweeps() == 200
    for fuel in (ELECTRICITY_KWH, GAS_KWH):
        series = residuals[fuel]
        assert len(series) == receipt["iterations"] == 200
        assert all(np.isfinite(v) and v >= 0 for v in series)
        # Converged: the last two sweeps agree to well inside the gate's
        # tolerance, and the walk does not end higher than it started.
        assert abs(series[-1] - series[-2]) < 1e-3
        assert series[-1] <= series[0] + 1e-12


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
            _TEST_PATHS.repository
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
    # Anthony's reviews of microcosm#1420: the recomputation shares the
    # stage's inputs, so the ONS-spend level is also held to a declared share
    # of the Energy Trends volume, and the QEP payment method, which that
    # share cannot see, to the gate's own declaration. Gas keeps Energy Trends.
    ratios = passed.details["ons_level_to_energy_trends_ratio"]
    assert set(ratios) == {ELECTRICITY_KWH}
    assert 0.92 < ratios[ELECTRICITY_KWH] < 0.93
    assert parameters["qep_payment_method"] == "all"
    with pytest.raises(ValueError, match="no qep_payment_method"):
        run({k: v for k, v in generous.items() if k != "qep_payment_method"})
    with pytest.raises(ValueError, match="qep_payment_method 'direct_debit' differs"):
        run({**generous, "qep_payment_method": "direct_debit"})
    failing(
        {
            **generous,
            "ons_level_to_energy_trends_ratio": {"minimum": 0.95, "maximum": 1.0},
        },
        match="ONS-spend level is 0.9258 of Energy Trends'",
    )
    with pytest.raises(ValueError, match="no ons_level_to_energy_trends_ratio"):
        run(
            {
                key: value
                for key, value in generous.items()
                if key != "ons_level_to_energy_trends_ratio"
            }
        )
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
        match="not the declared published level",
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
    # Convergence is checked, not asserted: a residual still moving over the
    # declared window is a truncated rake even inside the tolerance.
    assert parameters["convergence_window_sweeps"] == 10
    assert parameters["maximum_residual_change_over_window"] == 0.001
    assert set(passed.details["residual_change_over_window"]) == {
        ELECTRICITY_KWH,
        GAS_KWH,
    }
    assert max(passed.details["residual_change_over_window"].values()) < 0.001
    truncated = copy.deepcopy(receipt)
    series = truncated["sweep_residuals"][ELECTRICITY_KWH]
    series[-11] = series[-1] + 0.01
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": truncated},
        match="truncated, not converged",
    )
    # A spike inside the window fails too, with both endpoints untouched: the
    # gate reads the window's range, not its endpoints.
    spiked = copy.deepcopy(receipt)
    series = spiked["sweep_residuals"][GAS_KWH]
    series[-6] = series[-1] + 0.01
    assert series[-11] == receipt["sweep_residuals"][GAS_KWH][-11]
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": spiked},
        match="truncated, not converged",
    )
    # The window presupposes more declared sweeps than its own length.
    assert _declared_sweeps() > parameters["convergence_window_sweeps"]
    short = copy.deepcopy(receipt)
    short["sweep_residuals"][GAS_KWH] = short["sweep_residuals"][GAS_KWH][-10:]
    failing(
        generous,
        {"stage": "lcfs_consumption", "energy_rake": short},
        match="do not cover the 10-sweep convergence window",
    )
    failing(
        generous,
        {
            "stage": "lcfs_consumption",
            "energy_rake": {k: v for k, v in receipt.items() if k != "sweep_residuals"},
        },
        match="carries no sweep_residuals",
    )
    with pytest.raises(ValueError, match="convergence_window_sweeps"):
        run({**generous, "convergence_window_sweeps": 0})
    with pytest.raises(ValueError, match="declares no margins"):
        run({**generous, "margins": []})
    with pytest.raises(ValueError, match="differs from the stage"):
        run({**generous, "margins_period_value": 2023})


def test_the_recipient_rake_refuses_an_undeclared_sweep_count() -> None:
    """No silent default: the energy_rake window presumes the declared sweeps."""

    import dataclasses

    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    region_rake = next(
        operation
        for operation in stage.operations
        if operation.kind == "iterative_proportional_fit"
        and "region" in operation.parameters.get("margins", ())
    )
    without = dataclasses.replace(
        stage,
        operations=tuple(
            operation for operation in stage.operations if operation is not region_rake
        ),
    )
    with pytest.raises(ValueError, match="declared iterations are required"):
        _recipient_energy_rake_iterations(without)
    undeclared = dataclasses.replace(
        stage,
        operations=tuple(
            dataclasses.replace(
                operation,
                parameters={
                    key: value
                    for key, value in operation.parameters.items()
                    if key != "iterations"
                },
            )
            if operation is region_rake
            else operation
            for operation in stage.operations
        ),
    )
    with pytest.raises(ValueError, match="declared iterations are required"):
        _recipient_energy_rake_iterations(undeclared)


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


def test_cap_household_kwh_keeps_the_total_and_caps_every_household() -> None:
    # microcosm#1164: the kWh a capped household loses goes to the others in
    # proportion to their use, and a household lifted over is capped in turn.
    values = np.array([10.0, 4.0, 2.0, 1.0, 3.0])
    weights = np.array([1.0, 1.0, 2.0, 1.0, 2.0])
    population = np.array([True, True, True, True, False])
    capped, receipt = cap_household_kwh(
        values, weights=weights, population=population, ceiling=5.0
    )
    total = float(np.dot(weights[population], values[population]))
    assert float(np.dot(weights[population], capped[population])) == pytest.approx(
        total
    )
    assert (capped[population] <= 5.0 + 1e-9).all()
    assert capped[4] == 3.0  # outside the population, untouched
    assert receipt["ceiling_kwh"] == 5.0
    assert receipt["households_above_after"] == 0
    # One household was drawn above the ceiling; the rescaling lifts another
    # to it, which gives nothing up.
    assert receipt["households_capped"] == 1
    assert receipt["households_at_ceiling"] == 2
    assert receipt["capped_share_of_total"] == pytest.approx(5.0 / total)
    assert receipt["frame_kwh_after"] == pytest.approx(total)
    # Nothing to cap leaves the values as they were.
    same, untouched = cap_household_kwh(
        values, weights=weights, population=population, ceiling=50.0
    )
    np.testing.assert_array_equal(same, values)
    assert untouched["households_capped"] == 0 and untouched["rounds"] == 0
    # A frame too small for its level caps everyone and says the total fell
    # short; the energy_rake gate then refuses the receipt.
    short, infeasible = cap_household_kwh(
        np.array([10.0, 10.0]),
        weights=np.ones(2),
        population=np.ones(2, dtype=bool),
        ceiling=5.0,
    )
    assert short.tolist() == [5.0, 5.0]
    assert infeasible["feasible"] is False
    assert infeasible["frame_kwh_after"] == 10.0
    assert receipt["feasible"] is True


def test_cap_household_kwh_hands_the_excess_to_the_capped_households_cells() -> None:
    # microcosm#1164: with the rake's cells, what a capped household gives up
    # goes to the households below the ceiling in its own cells, so the cell
    # means stay at their targets where proportional handing-out moves them.
    band = np.array(["a", "a", "a", "b", "b", "b"])
    values = np.array([9.0, 2.0, 1.0, 2.0, 2.0, 2.0])
    weights = np.ones(6)
    population = np.ones(6, dtype=bool)
    cells = [(band, {"a": 4.0, "b": 2.0})]

    proportional, _ = cap_household_kwh(
        values, weights=weights, population=population, ceiling=6.0
    )
    raked, receipt = cap_household_kwh(
        values,
        weights=weights,
        population=population,
        ceiling=6.0,
        cells=cells,
        sweeps=20,
    )

    def mean(x, key):
        return float(x[band == key].mean())

    # Proportional handing-out moves both cells off their targets ...
    assert mean(proportional, "a") < 4.0 - 0.1 and mean(proportional, "b") > 2.0
    # ... the re-rake keeps both, with the capped household at the ceiling.
    assert mean(raked, "a") == pytest.approx(4.0)
    assert mean(raked, "b") == pytest.approx(2.0)
    assert raked[0] == 6.0 and (raked <= 6.0 + 1e-9).all()
    assert float(raked.sum()) == pytest.approx(float(values.sum()))
    assert receipt["households_capped"] == 1
    assert receipt["households_above_after"] == 0
    assert receipt["feasible"] is True


def test_declared_household_ceilings_are_the_domestic_meter_limits() -> None:
    ceilings = declared_household_kwh_ceilings(_declared())
    assert ceilings == {ELECTRICITY_KWH: 50_000.0, GAS_KWH: 73_200.0}
    assert declared_household_kwh_ceilings({}) is None
    for bad in ({"electricity": 50_000.0}, {"electricity": 50_000.0, "gas": 0.0}):
        with pytest.raises(ValueError, match="household_kwh_ceiling"):
            declared_household_kwh_ceilings({"household_kwh_ceiling": bad})


def test_rake_caps_after_the_level_and_keeps_the_published_totals() -> None:
    declared = _declared()
    margins = need_margins_from_facts(period_value=NEED_PERIOD)
    level, _ = published_energy_level(declared)
    n = 600
    rng = np.random.default_rng(11)
    table = pd.DataFrame(
        {
            ELECTRICITY_KWH: rng.uniform(1000.0, 5000.0, n),
            GAS_KWH: rng.uniform(5000.0, 20000.0, n),
        }
    )
    # Two outliers far above the domestic limits.
    table.loc[0, ELECTRICITY_KWH] = 400_000.0
    table.loc[1, GAS_KWH] = 900_000.0
    weights = np.full(n, 28_500_000.0 / n)
    raked, receipt = rake_energy_kwh(
        table,
        margins=margins,
        frs_region=np.array(["LONDON"] * n, dtype=object),
        income=rng.uniform(5e3, 2e5, n),
        weights=weights,
        iterations=1,
        level=level,
        ceilings=declared_household_kwh_ceilings(declared),
    )
    for fuel, ceiling in ((ELECTRICITY_KWH, 50_000.0), (GAS_KWH, 73_200.0)):
        values = raked[fuel].to_numpy(dtype=float)
        assert values.max() <= ceiling * (1 + 1e-9)
        assert float(np.dot(weights, values)) == pytest.approx(level[fuel], rel=1e-9)
        block = receipt["ceiling"][fuel]
        assert block["households_capped"] >= 1
        assert block["households_above_after"] == 0


def test_the_energy_rake_gate_checks_the_declared_household_ceilings() -> None:
    # microcosm#1164: the stage declares the domestic meter limits, so the gate
    # needs the ceiling block, no household above it and the total kept.
    from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

    gates = json.loads(
        (
            _TEST_PATHS.repository
            / "packages/microcosm-build/src/microcosm/build/uk/gates.json"
        ).read_text("utf-8")
    )
    by_id = {gate["id"]: gate for gate in gates["gates"]}
    parameters = {
        **by_id["uk_stage_lcfs_consumption_energy_rake"]["parameters"],
        "maximum_relative_deviation": 1.0,
    }
    _, _, _, _, receipt = _synthetic_rake_receipt()

    def run(energy_rake):
        return uk_stage_health_gate(
            evidence={"stage": "lcfs_consumption", "energy_rake": energy_rake},
            stage="lcfs_consumption",
            check="energy_rake",
            parameters=parameters,
        )

    passed = run(receipt)
    assert passed.passed, passed.failures
    assert passed.details["ceiling"][ELECTRICITY_KWH]["ceiling_kwh"] == 50_000.0
    assert passed.details["ceiling"][GAS_KWH]["ceiling_kwh"] == 73_200.0
    missing = copy.deepcopy(receipt)
    missing["ceiling"] = None
    assert any("no ceiling block" in f for f in run(missing).failures)
    above = copy.deepcopy(receipt)
    above["ceiling"][GAS_KWH]["households_above_after"] = 3
    assert any("above the declared" in f for f in run(above).failures)
    other = copy.deepcopy(receipt)
    other["ceiling"][ELECTRICITY_KWH]["ceiling_kwh"] = 25_000.0
    assert any("not the declared" in f for f in run(other).failures)
    lost = copy.deepcopy(receipt)
    lost["ceiling"][ELECTRICITY_KWH]["frame_kwh_after"] *= 0.9
    lost["ceiling"][ELECTRICITY_KWH]["feasible"] = False
    failures = run(lost).failures
    assert any("total after the cap" in f for f in failures)
    assert any("left below the electricity_kwh ceiling" in f for f in failures)


def test_the_stage_starts_only_the_mobile_home_cells_at_the_need_mean() -> None:
    # microcosm#1240: mobile homes are the one FRS dwelling type NEED has no
    # row for; the declaration names the all-dwellings mean they map to.
    declared = declared_zero_kwh_cells_at_need_mean(_declared())
    assert declared == {"accommodation": ("all_dwellings",)}
    for mapping in NEED_PROPERTY_TYPE_IDS_BY_GEOGRAPHY.values():
        assert mapping["MOBILE"] == "all_dwellings"
    assert declared_zero_kwh_cells_at_need_mean({}) is None
    for bad in (
        [],
        {},
        {"dwelling": ["all_dwellings"]},
        {"accommodation": "all_dwellings"},
        {"accommodation": []},
        {"accommodation": [""]},
    ):
        with pytest.raises(ValueError, match="zero_kwh_cells_at_need_mean"):
            declared_zero_kwh_cells_at_need_mean({"zero_kwh_cells_at_need_mean": bad})


def _mobile_home_table(n: int = 60):
    """England and Wales rows in one income band, every third a mobile home
    that drew no kWh (so the mobile-home cell is the only one with no kWh)."""

    rng = np.random.default_rng(17)
    accommodation = np.array(["HOUSE_DETACHED", "FLAT", "MOBILE"], dtype=object)[
        np.arange(n) % 3
    ]
    mobile = accommodation == "MOBILE"
    table = pd.DataFrame(
        {
            ELECTRICITY_KWH: np.where(mobile, 0.0, rng.uniform(1000.0, 5000.0, n)),
            GAS_KWH: np.where(mobile, 0.0, rng.uniform(5000.0, 20000.0, n)),
        }
    )
    arguments = {
        "margins": need_margins_from_facts(period_value=NEED_PERIOD),
        "frs_region": np.full(n, "LONDON", dtype=object),
        "income": rng.uniform(21e3, 29e3, n),
        "weights": rng.uniform(0.5, 2.0, n),
        "iterations": 50,
        "accommodation": accommodation,
        "gas_connected": ~mobile,
    }
    return table, mobile, arguments


def test_a_declared_cell_that_drew_no_kwh_starts_at_its_need_mean() -> None:
    table, mobile, arguments = _mobile_home_table()
    cell = f"{ENGLAND_AND_WALES_GEOGRAPHY_ID}:all_dwellings"
    # Undeclared, the cell has nothing to scale and stays at zero.
    plain, plain_receipt = rake_energy_kwh(table, **arguments)
    assert {
        (zero["margin"], zero["category"], zero["column"])
        for zero in plain_receipt["zero_current_cells"]
    } == {("_need_accommodation", cell, ELECTRICITY_KWH)}
    assert plain_receipt["cells_held_at_need_mean"] == []
    assert (plain.loc[mobile, ELECTRICITY_KWH] == 0.0).all()
    # Declared, its households start at the NEED mean and the rake fits it.
    raked, receipt = rake_energy_kwh(
        table,
        **arguments,
        zero_kwh_cells_at_need_mean={"accommodation": ("all_dwellings",)},
    )
    assert receipt["zero_current_cells"] == []
    assert receipt["cells_held_at_need_mean"] == [
        {"margin": "accommodation", "cell": cell, "column": ELECTRICITY_KWH}
    ]
    assert (raked.loc[mobile, ELECTRICITY_KWH] > 0.0).all()
    fitted = receipt["fit"]["accommodation"]["cells"][cell][ELECTRICITY_KWH]
    assert abs(fitted["relative_deviation"]) < 0.05
    # The mobile homes have no gas connection, so gas has no cell to start
    # and its rake runs as it did undeclared.
    assert (raked.loc[mobile, GAS_KWH] == 0.0).all()
    assert (
        receipt["sweep_residuals"][GAS_KWH] == plain_receipt["sweep_residuals"][GAS_KWH]
    )
    # A cell with one positive row is left to the rake.
    drawn = table.copy()
    drawn.loc[np.flatnonzero(mobile)[0], ELECTRICITY_KWH] = 500.0
    _, kept = rake_energy_kwh(
        drawn,
        **arguments,
        zero_kwh_cells_at_need_mean={"accommodation": ("all_dwellings",)},
    )
    assert kept["cells_held_at_need_mean"] == []
    assert kept["zero_current_cells"] == []


def test_a_held_cell_must_be_a_published_cell_of_a_swept_margin() -> None:
    table, _, arguments = _mobile_home_table()
    with pytest.raises(ValueError, match="NEED publishes no accommodation cell"):
        rake_energy_kwh(
            table,
            **arguments,
            zero_kwh_cells_at_need_mean={"accommodation": ("park_homes",)},
        )
    with pytest.raises(ValueError, match="does not sweep"):
        rake_energy_kwh(
            table,
            **arguments,
            zero_kwh_cells_at_need_mean={"tenure": ("owner_occupied",)},
        )


def test_the_energy_rake_gate_takes_only_the_declared_held_cells() -> None:
    from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

    gates = json.loads(
        (
            _TEST_PATHS.repository
            / "packages/microcosm-build/src/microcosm/build/uk/gates.json"
        ).read_text("utf-8")
    )
    by_id = {gate["id"]: gate for gate in gates["gates"]}
    parameters = {
        **by_id["uk_stage_lcfs_consumption_energy_rake"]["parameters"],
        "maximum_relative_deviation": 1.0,
    }
    _, _, _, _, receipt = _synthetic_rake_receipt()

    def run(energy_rake):
        return uk_stage_health_gate(
            evidence={"stage": "lcfs_consumption", "energy_rake": energy_rake},
            stage="lcfs_consumption",
            check="energy_rake",
            parameters=parameters,
        )

    passed = run(receipt)
    assert passed.passed, passed.failures
    assert passed.details["cells_held_at_need_mean"] == 0
    held = copy.deepcopy(receipt)
    held["cells_held_at_need_mean"] = [
        {
            "margin": "accommodation",
            "cell": f"{SCOTLAND_GEOGRAPHY_ID}:all_dwellings",
            "column": ELECTRICITY_KWH,
        }
    ]
    declared = run(held)
    assert declared.passed, declared.failures
    # The details count the cells and never name one.
    assert declared.details["cells_held_at_need_mean"] == 1
    assert "all_dwellings" not in json.dumps(declared.details)
    undeclared = copy.deepcopy(receipt)
    undeclared["cells_held_at_need_mean"] = [
        {
            "margin": "accommodation",
            "cell": f"{ENGLAND_AND_WALES_GEOGRAPHY_ID}:detached",
            "column": ELECTRICITY_KWH,
        }
    ]
    assert any("does not declare" in f for f in run(undeclared).failures)
    missing = {k: v for k, v in receipt.items() if k != "cells_held_at_need_mean"}
    assert any("carries no cells_held_at_need_mean" in f for f in run(missing).failures)


def test_the_lcfs_stage_key_covers_energy_pricing() -> None:
    """The independent review of microcosm#1420, finding 5: the stage's energy
    level and prices live in energy_pricing, so a code-only change there must
    re-key the stage rather than resume a cached pass."""

    from microcosm.build.uk_runtime import energy_pricing
    from microcosm.build.uk_runtime.graph_kernels import _STAGE_HELPER_MODULES

    assert energy_pricing in _STAGE_HELPER_MODULES["lcfs_consumption"]


def test_each_bound_ons_energy_row_is_a_fuel_levelled_to_ons_spending() -> None:
    """The independent review of microcosm#1420, finding 11: a per-fuel ONS
    spending row binds only when the stage levels that fuel to ONS spending.
    Otherwise the solve pulls a frame levelled to Energy Trends onto ONS's
    level, the pull microcosm#1113 took out (D1). Gas's row is held out while
    ONS 04.5.2 counts bottled gas, so rebinding it (microcosm#1095) needs gas
    in spend_level_fuels too."""

    from importlib import resources as importlib_resources

    from microcosm.build.uk_runtime.energy_pricing import (
        ELECTRICITY_FUEL,
        GAS_FUEL,
        ONS_HOUSEHOLD_SPEND_CONCEPTS,
        declared_spend_level_fuels,
    )
    from microcosm.build.uk_runtime.measure_simulation import (
        load_uk_calibration_measure_exclusions,
    )

    def load(resource: str) -> dict:
        return json.loads(
            importlib_resources.files("microcosm.build.uk")
            .joinpath(resource)
            .read_text(encoding="utf-8")
        )

    fuel_of_concept = {
        ONS_HOUSEHOLD_SPEND_CONCEPTS[ELECTRICITY_KWH]: ELECTRICITY_FUEL,
        ONS_HOUSEHOLD_SPEND_CONCEPTS[GAS_KWH]: GAS_FUEL,
    }
    rows = {}
    for target in load("uk_population_targets.json")["targets"]:
        concept = (target.get("ledger_selector") or {}).get("source_concept")
        if isinstance(concept, str) and concept in fuel_of_concept:
            rows[target["target_id"]] = fuel_of_concept[concept]
    assert sorted(rows.values()) == [ELECTRICITY_FUEL, GAS_FUEL]
    held = {entry["name"] for entry in load_uk_calibration_measure_exclusions()} | {
        entry["target_id"]
        for entry in load("target_reference_signed_exclusions.json")["exclusions"]
        if "row" not in entry
    }
    bound = {
        target_id: fuel for target_id, fuel in rows.items() if target_id not in held
    }
    assert bound == {"ons.household_electricity_expenditure": ELECTRICITY_FUEL}
    assert set(bound.values()) <= set(declared_spend_level_fuels(_declared()))


def test_the_ons_row_variant_is_defined_once() -> None:
    """docs/shared-constants.md: the lcfs_consumption stage reads the ONS
    resource and row variant energy_pricing defines (the independent review of
    microcosm#1420), and the energy view only adds the domestic concept."""

    from microcosm.build.uk_runtime import energy_pricing, lcfs_consumption

    assert lcfs_consumption.UK_ONS_HFCE_ROW_DIMENSIONS is (
        energy_pricing.UK_ONS_HFCE_ROW_DIMENSIONS
    )
    assert lcfs_consumption.UK_LCFS_ROAD_FUEL_LEVEL_RESOURCE == (
        energy_pricing.UK_ONS_HOUSEHOLD_EXPENDITURE_RESOURCE
    )
    assert dict(energy_pricing.ONS_HOUSEHOLD_SPEND_DIMENSIONS) == {
        "consumption_concept": "domestic",
        **energy_pricing.UK_ONS_HFCE_ROW_DIMENSIONS,
    }
