"""M2 (energy) and M3 (road fuel) desk measurements for the VAT/fuel/energy plan.

Reads only existing artefacts: the pinned Chronicle feed (825406f, rebuilt
consumer_facts.jsonl), Microcosm's vendored resources, the 2026-09-30 spine and
national H5s, and the LCFS 2023-24 household file. No solve, no engine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PE = Path("/Users/mariajuaristi/Desktop/PolicyEngine")
OUT = Path(__file__).resolve().parent
FEED = (
    PE
    / "data/ukds/acceptance/chronicle-uk-artifact-825406f-rebuilt/consumer_facts.jsonl"
)
RES = (
    PE / "repos/populace-national-main/packages/microcosm-build/src/microcosm/build/uk"
)
SPINE = (
    PE
    / "data/ukds/acceptance/national-main-20260930/spine-main-c5a1cba8/spine-main-c5a1cba8.h5"
)
FINAL = PE / "runs/uk-national-main-c5a1cba8/microcosm_uk_2024_25.h5"
LCFS = PE / "data/ukds/lcfs_2023_24/dvhh_ukanon_v2_2023.tab"
WEEKS = 365.25 / 7


def feed_consumer_trends() -> dict[str, dict[str, float]]:
    """Annual and quarterly ONS Consumer Trends values (GBP) from the pinned feed."""
    pattern = re.compile(r"ons\.consumer_trends\.(04cn|07cn_07_2_2)[^\"]*")
    out: dict[str, dict[str, float]] = {}
    with FEED.open() as handle:
        for line in handle:
            if "ons.consumer_trends." not in line:
                continue
            fact = json.loads(line)
            rid = str((fact.get("lineage") or {}).get("source_record_id", ""))
            if not pattern.search(rid):
                continue
            coicop = (fact.get("dimensions") or {}).get("coicop")
            period = fact["period"]
            key = str(period["value"])
            out.setdefault(str(coicop), {})[key] = float(fact["value"])
    return out


def fiscal_sum(series: dict[str, float], start: int) -> float | None:
    keys = [f"{start}-Q2", f"{start}-Q3", f"{start}-Q4", f"{start + 1}-Q1"]
    alt = [f"{start}_q2", f"{start}_q3", f"{start}_q4", f"{start + 1}_q1"]
    for candidate in (keys, alt):
        if all(k in series for k in candidate):
            return sum(series[k] for k in candidate)
    return None


def vendored(name: str) -> list[dict]:
    return json.loads((RES / name).read_text())["rows"]


def main() -> None:
    result: dict = {}
    ct = feed_consumer_trends()
    result["feed_period_keys_sample"] = {
        k: sorted(v)[:3] + sorted(v)[-3:] for k, v in ct.items()
    }

    # --- M2 energy -------------------------------------------------------
    desnz = vendored("desnz_domestic_energy_facts.json")

    def trends(fuel: str, quarters: list[str]) -> float:
        concept = f"desnz.energy_trends.domestic_{fuel}_consumption"
        rows = {
            str(r["period"]["value"]): float(r["value"])
            for r in desnz
            if r["concept"] == concept
            and r["period"]["type"] == "quarter"
            and (r.get("dimensions") or {}).get("temperature_adjustment")
            == "actual_temperature"
        }
        return sum(rows[q] for q in quarters) / 1e9

    fy2024 = ["2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1"]
    cy2025 = ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"]
    meters = {
        (r["concept"], r["geography"]["id"]): float(r["value"])
        for r in desnz
        if r["concept"].endswith("meter_count") and r["period"]["value"] == 2024
    }
    qep = vendored("qep_energy_prices.json")

    def qep_value(concept: str, ptype: str, pval: int) -> float:
        rows = [
            r
            for r in qep
            if r["concept"] == concept
            and r["period"]["type"] == ptype
            and int(r["period"]["value"]) == pval
            and r["geography"]["id"] == "K02000001"
            and (r.get("dimensions") or {}).get("payment_method") == "all"
        ]
        assert len(rows) == 1, (concept, ptype, pval, len(rows))
        return float(rows[0]["value"])

    spine = pd.read_hdf(SPINE, "household")
    w = spine.household_weight.to_numpy()
    region = np.array(
        [
            x.decode() if isinstance(x, bytes) else getattr(x, "name", str(x))
            for x in spine.region
        ]
    )
    ni = region == "NORTHERN_IRELAND"
    households = {
        "uk": float(w.sum()),
        "gb": float(w[~ni].sum()),
        "ni": float(w[ni].sum()),
    }
    gas_connected = float(w[spine.gas_consumption.to_numpy() > 0].sum())
    energy = {
        "households_design_m": {k: v / 1e6 for k, v in households.items()},
        "gas_connected_households_design_m": gas_connected / 1e6,
    }
    energy["gb_meters_2024_m"] = {
        "electricity": meters[
            ("desnz.subnational.domestic_electricity_meter_count", "K03000001")
        ]
        / 1e6,
        "gas": meters[("desnz.subnational.domestic_gas_meter_count", "K03000001")]
        / 1e6,
    }
    for label, quarters, ptype, pval, ons_key in (
        ("fy2024_25", fy2024, "fiscal_year", 2024, None),
        ("cy2025", cy2025, "calendar_year", 2025, "2025"),
    ):
        block = {}
        for fuel, coicop, n_fixed in (
            ("electricity", "04.5.1", households["uk"]),
            ("gas", "04.5.2", gas_connected),
        ):
            volume = trends(fuel, quarters)
            unit = qep_value(
                f"desnz.qep.domestic_{fuel}_variable_unit_cost", ptype, pval
            )
            fixed = qep_value(f"desnz.qep.domestic_{fuel}_fixed_cost", ptype, pval)
            series = ct.get(coicop, {})
            ons = series.get(ons_key) if ons_key else fiscal_sum(series, 2024)
            unit_only = volume * unit
            with_fixed = unit_only + fixed * n_fixed / 1e9
            block[fuel] = {
                "energy_trends_twh": volume,
                "qep_unit_gbp_per_kwh": unit,
                "qep_fixed_gbp_per_year": fixed,
                "unit_only_bn": unit_only,
                "unit_plus_fixed_per_household_bn": with_fixed,
                "ons_bn": None if ons is None else ons / 1e9,
                "ons_over_unit_plus_fixed": None
                if ons is None
                else ons / 1e9 / with_fixed,
                "ons_over_unit_only": None if ons is None else ons / 1e9 / unit_only,
                "ons_implied_all_in_pence_per_kwh": None
                if ons is None
                else 100 * ons / 1e9 / volume,
                "qep_all_in_pence_per_kwh": 100 * with_fixed / volume,
            }
        energy[label] = block
    # LCFS FYE2024 energy against ONS 04.5 FY2023-24
    lcfs = pd.read_csv(
        LCFS,
        sep="\t",
        usecols=lambda c: (
            c.lower()
            in {"p537", "p604", "weighta", "c72211", "c72212", "c72213", "a124"}
        ),
        low_memory=False,
    )
    lcfs.columns = [c.lower() for c in lcfs.columns]
    for column in lcfs.columns:
        lcfs[column] = pd.to_numeric(
            lcfs[column].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
        ).fillna(0.0)
    lw = lcfs.weighta * 1000
    ons_45_fy2023 = fiscal_sum(ct.get("04.5", {}), 2023)
    energy["lcfs_fye2024_p537_bn"] = float((lcfs.p537 * lw).sum() * WEEKS / 1e9)
    energy["ons_04_5_fy2023_24_bn"] = (
        None if ons_45_fy2023 is None else ons_45_fy2023 / 1e9
    )
    result["m2_energy"] = energy

    # --- M3 road fuel ------------------------------------------------------
    fuel = {}
    s722 = ct.get("07.2.2", {})
    fuel["ons_07_2_2_bn"] = {
        "cy2024": s722.get("2024", 0) / 1e9,
        "fy2024_25": (fiscal_sum(s722, 2024) or 0) / 1e9,
        "cy2025": s722.get("2025", 0) / 1e9,
    }
    other_share = float(
        (lcfs.c72213 * lw).sum()
        / ((lcfs.c72211 + lcfs.c72212 + lcfs.c72213) * lw).sum()
    )
    fuel["lcfs_other_fuels_share"] = other_share
    petrol = float(np.dot(w, spine.petrol_spending))
    diesel = float(np.dot(w, spine.diesel_spending))
    level = fuel["ons_07_2_2_bn"]["cy2024"] * (1 - other_share)
    factor = level / ((petrol + diesel) / 1e9)
    fuel["frame_design_bn"] = {"petrol": petrol / 1e9, "diesel": diesel / 1e9}
    fuel["level_bn"] = level
    fuel["level_factor"] = factor
    anchors = vendored("road_fuel_anchors.json")

    def anchor(concept: str, pval: int) -> float:
        rows = [
            r
            for r in anchors
            if r["concept"] == concept
            and r["period"]["type"] == "calendar_year"
            and int(r["period"]["value"]) == pval
        ]
        assert len(rows) == 1, (concept, pval, len(rows))
        return float(rows[0]["value"])

    p_ulsp, p_ulsd = (
        anchor("desnz.road_fuel.annual_ulsp_pump_price", 2024) / 100,
        anchor("desnz.road_fuel.annual_ulsd_pump_price", 2024) / 100,
    )
    growth_2025 = (
        1 - 0.0269132439679719
    )  # pe-uk petrol/diesel_spending_litre_proxy 2025
    duty = 0.5295
    for label, (pp, pd_) in {
        "rac_divisors_pe_uk_2_100": (1.44, 1.52),
        "desnz_2024_divisors": (p_ulsp, p_ulsd),
    }.items():
        litres_2024 = (petrol * factor / pp + diesel * factor / pd_) / 1e9
        duty_2025 = litres_2024 * growth_2025 * duty
        fuel[label] = {
            "litres_2024_bn": litres_2024,
            "fuel_duty_2025_bn": duty_2025,
            "vs_obr_cars_14_4": duty_2025 / 14.4 - 1,
        }
    flag = spine.has_fuel_consumption.astype(bool).to_numpy()
    total_fuel = (spine.petrol_spending + spine.diesel_spending).to_numpy()
    fuel["fuel_car_households_design_m"] = float(w[flag].sum() / 1e6)
    fuel["fuel_car_zero_fuel_share_design"] = float(
        w[flag & (total_fuel <= 0)].sum() / w[flag].sum()
    )
    vehicles = lcfs.a124 > 0
    fuel["lcfs_vehicle_households_zero_diary_fuel_share"] = float(
        lw[vehicles & ((lcfs.c72211 + lcfs.c72212) <= 0)].sum() / lw[vehicles].sum()
    )
    result["m3_fuel"] = fuel
    (OUT / "m2_m3_desk.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
