"""FY2024-25 Ofgem cap pricing and NEED kWh margins from vendored facts (#890 F)."""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def _synthetic_rake_receipt():
    from microcosm.build.uk_runtime.energy_pricing import (
        ELECTRICITY_KWH,
        GAS_KWH,
        rake_energy_kwh,
    )

    margins = need_margins_from_facts()
    n = 120
    rng = np.random.default_rng(5)
    table = pd.DataFrame(
        {
            ELECTRICITY_KWH: rng.uniform(1000.0, 5000.0, n),
            GAS_KWH: np.where(
                np.arange(n) % 5 == 0, 0.0, rng.uniform(5000.0, 20000.0, n)
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
    _, receipt = rake_energy_kwh(
        table,
        margins=margins,
        frs_region=region,
        income=income,
        weights=rng.uniform(0.5, 2.0, n),
        iterations=50,
        tenure=tenure,
        accommodation=accommodation,
        use_region_margin=True,
        gas_connected=table[GAS_KWH].to_numpy() > 0,
    )
    return margins, receipt


def test_rake_fit_targets_lockstep_with_the_vendored_need_rows() -> None:
    """Every target in the fit block is the vendored NEED mean, recomputed here."""

    from microcosm.build.uk_runtime.energy_pricing import ELECTRICITY_KWH, GAS_KWH

    margins, receipt = _synthetic_rake_receipt()
    fit = receipt["fit"]
    assert set(fit) == {"income", "tenure", "accommodation", "region"}
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
                assert cell[fuel]["target"] == pytest.approx(rows[fuel]), (margin, key)
            checked += 1
    assert checked >= 10
    # Region is swept last, so it fits exactly; gas is averaged over its
    # connected rows, a smaller weight than electricity's.
    assert fit["region"]["max_abs_relative_deviation"]["electricity_kwh"] < 1e-9
    assert fit["region"]["max_abs_relative_deviation"]["gas_kwh"] < 1e-9
    london = fit["region"]["cells"][f"{ENGLAND_AND_WALES_GEOGRAPHY_ID}:london"]
    assert (
        london["gas_kwh"]["weighted_rows"] < london["electricity_kwh"]["weighted_rows"]
    )


def test_need_is_checked_at_stage_time_by_the_energy_rake_gate() -> None:
    """The NEED kWh fit is a stage-health gate on the lcfs receipt (#890, ruling 2026-09-15).

    The two NEED mean-spend anchors no longer sit in uk_aggregate_admin (the
    calibrated frame is held to ONS 04.5). The gate recomputes every cell
    target from the vendored facts and holds the IPF residual to one
    declared tolerance.
    """

    import copy
    import json
    from pathlib import Path

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
    assert "-24 %" in by_id["uk_aggregate_admin"]["notes"]
    parameters = by_id["uk_stage_lcfs_consumption_energy_rake"]["parameters"]
    assert parameters["check"] == "energy_rake"
    assert by_id["uk_stage_lcfs_consumption_energy_rake"]["evidence_absent_blocks"]
    _, receipt = _synthetic_rake_receipt()
    evidence = {"stage": "lcfs_consumption", "energy_rake": receipt}

    def run(params=parameters, ev=evidence):
        return uk_stage_health_gate(
            evidence=ev,
            stage="lcfs_consumption",
            check="energy_rake",
            parameters=params,
        )

    generous = {**parameters, "maximum_relative_deviation": 1.0}
    passed = run(generous)
    assert passed.passed, passed.failures
    assert passed.details["worst"]["region:gas_kwh"] < 1e-9
    assert passed.details["cells_fact_checked"] > 0
    # Residual tolerance breached.
    failed = run({**parameters, "maximum_relative_deviation": 0.0})
    assert not failed.passed and any("residual tolerance" in f for f in failed.failures)
    # A margin the receipt raked but the gate did not declare.
    result = run({**generous, "margins": ["income", "tenure", "accommodation"]})
    assert not result.passed and any("undeclared margins" in f for f in result.failures)
    # A declared margin the receipt did not rake.
    trimmed = copy.deepcopy(receipt)
    trimmed["margins"] = ["income", "tenure", "accommodation"]
    del trimmed["fit"]["region"]
    result = run(generous, {"stage": "lcfs_consumption", "energy_rake": trimmed})
    assert not result.passed and any("was not raked" in f for f in result.failures)
    # A cell raked to something other than the vendored NEED mean.
    tampered = copy.deepcopy(receipt)
    key = next(iter(tampered["fit"]["income"]["cells"]))
    tampered["fit"]["income"]["cells"][key]["gas_kwh"]["target"] *= 1.01
    result = run(generous, {"stage": "lcfs_consumption", "energy_rake": tampered})
    assert not result.passed and any(
        "not the vendored NEED mean" in f for f in result.failures
    )
    # Zero-current cells, the wrong gas population, and a missing fit block.
    zeroed = {
        **receipt,
        "zero_current_cells": [{"margin": "_need_income", "category": "x"}],
    }
    result = run(generous, {"stage": "lcfs_consumption", "energy_rake": zeroed})
    assert not result.passed and any("zero current mean" in f for f in result.failures)
    result = run(
        generous,
        {
            "stage": "lcfs_consumption",
            "energy_rake": {**receipt, "gas_rake_population": "all_rows"},
        },
    )
    assert not result.passed and any("gas_connected_rows" in f for f in result.failures)
    result = run(
        generous,
        {
            "stage": "lcfs_consumption",
            "energy_rake": {k: v for k, v in receipt.items() if k != "fit"},
        },
    )
    assert not result.passed and any("no fit block" in f for f in result.failures)
    with pytest.raises(ValueError, match="declares no margins"):
        run({**generous, "margins": []})


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


def test_gas_is_raked_over_connected_rows_and_electricity_over_all() -> None:
    from microcosm.build.uk_runtime.energy_pricing import (
        ELECTRICITY_KWH,
        GAS_KWH,
        rake_energy_kwh,
    )

    margins = need_margins_from_facts()
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
    assert raked[ELECTRICITY_KWH].mean() == pytest.approx(cell["electricity_kwh"])
    # Gas: the connected rows carry the NEED mean; unconnected rows stay zero.
    assert raked.loc[connected, GAS_KWH].mean() == pytest.approx(cell["gas_kwh"])
    assert (raked.loc[~connected, GAS_KWH] == 0.0).all()
    assert raked[GAS_KWH].mean() == pytest.approx(cell["gas_kwh"] * connected.mean())
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
    assert unmasked[GAS_KWH].mean() == pytest.approx(cell["gas_kwh"])
