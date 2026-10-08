from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.lcfs_consumption import (
    BUS_FARE_LCFS_CODES,
    LCFS_ACCOMM_MAP,
    LCFS_TENURE_MAP,
    UK_LCFS_CONSUMPTION_PREDICTORS,
    UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
    UKLCFSConsumptionResult,
    UKLCFSConsumptionStageTransform,
    assign_recipient_has_fuel,
    clean_lcfs_consumption_table,
    derive_energy_from_lcfs,
    recipient_predictors,
    support_clip_to_donor,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame


def _household() -> pd.DataFrame:
    base = {
        "case": [1, 2, 3, 4],
        "g018": [2, 1, 3, 1],
        "g019": [0, 1, 2, 0],
        "gorx": [7, 12, 10, 1],
        "a124": [0, 1, 2, 9],
        "p389p": [100.0, 200.0, 300.0, 400.0],
        "p344p": [150.0, 250.0, 350.0, 450.0],
        "weighta": [1.5, 2.0, 2.5, 3.0],
        "a122": [4, 8, 5, 7],
        "a121": [4, 5, 6, 7],
        "b226": [6.0, 0.0, 0.0, 0.0],
        "b489": [0.0, 9.0, 8.0, 0.0],
        "b490": [0.0, 4.0, 0.0, 0.0],
        "p537": [10.0, 20.0, 30.0, -1.0],
    }
    for source in (
        "p601",
        "p602",
        "p603",
        "p604",
        "p605",
        "p606",
        "p607",
        "p608",
        "p609",
        "p610",
        "p611",
        "p612",
        "c72211",
        "c72212",
        *BUS_FARE_LCFS_CODES,
    ):
        base[source] = [1.0, 2.0, 3.0, 4.0]
    return pd.DataFrame(base)


def _person() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [1, 1, 3],
            "b303p": [10.0, 5.0, 7.0],
            "b3262p": [1.0, 2.0, 3.0],
            "p049p": [4.0, 5.0, 6.0],
        }
    )


def test_lcfs_donor_cleaning_arithmetic_and_lossy_maps() -> None:
    donor = clean_lcfs_consumption_table(_person(), _household())

    assert donor["region"].tolist() == [
        "LONDON",
        "NORTHERN_IRELAND",
        "WALES",
        "NORTH_EAST",
    ]
    assert LCFS_TENURE_MAP[4] == "RENT_PRIVATELY"
    assert LCFS_TENURE_MAP[8] == "RENT_PRIVATELY"
    assert LCFS_ACCOMM_MAP[4] == "FLAT"
    assert LCFS_ACCOMM_MAP[5] == "FLAT"
    assert donor["household_weight"].tolist() == [1500.0, 2000.0, 2500.0, 3000.0]
    # LCFS a124 is the vehicle-count predictor, clipped to the declared 0-5.
    assert donor["num_vehicles"].tolist() == [0, 1, 2, 5]
    assert "has_fuel_consumption" not in donor
    assert UK_LCFS_CONSUMPTION_PREDICTORS[-1] == "num_vehicles"
    assert "has_fuel_consumption" not in UK_LCFS_CONSUMPTION_PREDICTORS
    assert np.isclose(
        donor.loc[0, "employment_income"],
        (10.0 + 5.0) * (365.25 / 7),
    )
    assert donor.loc[1, "employment_income"] == 0.0
    assert np.isclose(donor.loc[0, "bus_fare_spending"], 3.0 * (365.25 / 7))


def test_energy_split_exercises_four_cases_fallback_and_clamp() -> None:
    household = _household()
    split = derive_energy_from_lcfs(household)

    assert split["electricity_consumption"].tolist() == [6.0, 5.0, 4.8, 0.0]
    assert split["gas_consumption"].tolist() == [4.0, 4.0, 3.2, 0.0]

    fallback = household.copy()
    fallback["b226"] = 0.0
    fallback["b489"] = 0.0
    fallback["p537"] = [10.0, 20.0, 30.0, 40.0]

    split = derive_energy_from_lcfs(fallback)

    np.testing.assert_allclose(
        split["electricity_consumption"], [5.2, 10.4, 15.6, 20.8]
    )
    np.testing.assert_allclose(split["gas_consumption"], [4.8, 9.6, 14.4, 19.2])


def test_recipient_has_fuel_is_conditioned_on_vehicle_count_and_deterministic() -> None:
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_household_id": [10, 20],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame(
            {
                "household_id": [10, 20],
                "household_weight": [1.0, 1.0],
                "num_vehicles": [0, 2],
            }
        ),
        time_period="2023",
    )

    first = assign_recipient_has_fuel(frame, rate=1.0, seed=0)
    second = assign_recipient_has_fuel(frame, rate=1.0, seed=0)

    assert first.tolist() == [False, True]
    assert second.tolist() == first.tolist()


def test_negative_donor_consumption_is_floored_with_a_receipt() -> None:
    """microcosm#1063 c9: a diary spend netting refunds below zero is raised to
    the declared floor before the clip ranges and the imputation see it."""
    from microcosm.build.uk_runtime.lcfs_consumption import (
        CONSUMPTION_VARIABLE_RENAMES,
        floor_negative_donor_consumption,
    )

    columns = list(CONSUMPTION_VARIABLE_RENAMES.values())
    donor = pd.DataFrame({column: [10.0, 20.0, 30.0] for column in columns})
    donor.loc[1, "housing_water_and_electricity_consumption"] = -14_165.0
    donor.loc[2, "housing_water_and_electricity_consumption"] = -1.0

    floored, receipt = floor_negative_donor_consumption(donor)

    assert floored["housing_water_and_electricity_consumption"].tolist() == [
        10.0,
        0.0,
        0.0,
    ]
    for column in columns:
        if column != "housing_water_and_electricity_consumption":
            assert floored[column].tolist() == [10.0, 20.0, 30.0]
    assert receipt["floor"] == 0.0
    assert receipt["rows_raised"] == 2
    assert receipt["remaining_negative_rows"] == 0
    housing = receipt["columns"]["housing_water_and_electricity_consumption"]
    assert housing == {
        "rows_raised": 2,
        "negative_mass": -14_166.0,
        "minimum_before": -14_165.0,
    }
    assert receipt["columns"]["food_and_non_alcoholic_beverages_consumption"] == {
        "rows_raised": 0,
        "negative_mass": 0.0,
        "minimum_before": 10.0,
    }
    with pytest.raises(ValueError, match="missing consumption column"):
        floor_negative_donor_consumption(donor.drop(columns=[columns[0]]))


def test_support_clip_exempts_raked_energy_columns() -> None:
    donor = pd.DataFrame(
        {column: [1.0, 5.0] for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS}
    )
    draws = pd.DataFrame(
        {column: [0.0, 10.0] for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS}
    )

    clip_result = support_clip_to_donor(
        draws,
        donor,
        exempt={
            "electricity_consumption",
            "gas_consumption",
            "domestic_energy_consumption",
        },
    )
    clipped = clip_result.clipped

    assert clipped["food_and_non_alcoholic_beverages_consumption"].tolist() == [
        1.0,
        5.0,
    ]
    assert clipped["electricity_consumption"].tolist() == [0.0, 10.0]
    receipt = clip_result.receipt.evidence()["columns"]
    assert receipt["food_and_non_alcoholic_beverages_consumption"] == {
        "donor_min": 1.0,
        "donor_max": 5.0,
        "clipped_low_rows": 1,
        "clipped_high_rows": 1,
        "rows_considered": 2,
    }
    assert receipt["electricity_consumption"] == {
        "exempt": True,
        "rows_considered": 2,
    }
    assert receipt["gas_consumption"] == {
        "exempt": True,
        "rows_considered": 2,
    }
    assert receipt["domestic_energy_consumption"] == {
        "exempt": True,
        "rows_considered": 2,
    }
    transform = UKLCFSConsumptionStageTransform(stage=object(), engine=object())
    transform.last_result = UKLCFSConsumptionResult(
        frame=object(),
        support_clip=clip_result.receipt,
    )
    assert transform.checkpoint_metadata()["evidence"] == {
        "stage": "lcfs_consumption",
        "support_clip": clip_result.receipt.evidence(),
    }


def test_declared_vehicle_count_mapping_and_clip_lockstep_with_the_module() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        HOUSEHOLD_LCFS_RENAMES,
        UK_LCFS_VEHICLE_COUNT_RANGE,
    )

    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    derive = stage.operations[0]
    assert derive.kind == "derive"
    assert derive.parameters["mappings"] == {"num_vehicles": "a124"}
    assert HOUSEHOLD_LCFS_RENAMES["a124"] == "num_vehicles"
    assert tuple(derive.parameters["clip"]["num_vehicles"]) == (
        UK_LCFS_VEHICLE_COUNT_RANGE
    )
    assert {artifact["role"] for artifact in stage.artifacts} >= {
        "road_fuel_anchors",
        "licensed_cars_fuel_type",
    }
    assert not any(
        artifact["role"] == "was_bridge_donor" for artifact in stage.artifacts
    )


def test_ice_share_comes_from_the_vendored_veh1103_stock() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        ice_share_from_licensed_cars,
        lcfs_ice_share,
    )
    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    rate, receipt = lcfs_ice_share(stage)

    def licensed(fuel_type: str) -> float:
        rows = vendored_rows(
            "licensed_cars_fuel_type.json",
            period_type="calendar_year",
            period_value=2024,
            geography_id="K03000001",
            dimensions={"fuel_type": fuel_type},
        )
        assert len(rows) == 1
        return float(rows[0]["value"])

    expected = 1.0 - (
        licensed("battery_electric") + licensed("fuel_cell_electric")
    ) / licensed("all")
    assert rate == expected
    assert 0.95 < rate < 0.97
    assert receipt["rule"] == "one_minus_zero_emission_share"
    assert receipt["period_value"] == 2024
    assert receipt["geography_id"] == "K03000001"
    assert set(receipt["zero_emission_licensed_cars"]) == {
        "battery_electric",
        "fuel_cell_electric",
    }
    assert receipt["licensed_cars"] == licensed("all")
    assert len(receipt["source_record_ids"]) == 3

    declared = {
        op.kind: dict(op.parameters)
        for op in stage.operations
        if op.kind == "assign_binary_from_rate"
    }["assign_binary_from_rate"]
    foreign = {**declared, "rate_resource": "need_energy_facts.json"}
    with pytest.raises(ValueError, match="licensed_cars_fuel_type.json"):
        ice_share_from_licensed_cars(foreign)
    with pytest.raises(ValueError, match="rate_rule"):
        ice_share_from_licensed_cars({**declared, "rate_rule": "share"})
    with pytest.raises(ValueError, match="zero_emission_fuel_types"):
        ice_share_from_licensed_cars({**declared, "zero_emission_fuel_types": []})
    with pytest.raises(ValueError, match="expected one"):
        ice_share_from_licensed_cars({**declared, "period_value": 1999})


def test_recipient_predictors_read_the_vehicle_count_numerically() -> None:
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS,
        recipient_predictors,
    )

    class _Engine:
        def materialize(self, frame, variables, period):
            n = len(frame.table("household"))
            return {name: np.full(n, 1.0) for name in variables}

        def variable_metadata(self, name):
            return SimpleNamespace(entity="household")

    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_benunit_id": [1, 2, 3],
                "person_household_id": [10, 20, 30],
                "age": [40.0, 35.0, 70.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2, 3]}),
        household=pd.DataFrame(
            {
                "household_id": [10, 20, 30],
                "household_weight": [1.0, 1.0, 1.0],
                "region": ["LONDON", "WALES", "SCOTLAND"],
                "tenure_type": ["OWNED_OUTRIGHT", "RENT_PRIVATELY", "OWNED_OUTRIGHT"],
                "accommodation_type": ["FLAT", "HOUSE_DETACHED", "FLAT"],
                "num_vehicles": [0, 2.6, 11],
            }
        ),
        time_period="2024",
    )

    predictors = recipient_predictors(frame, _Engine())

    assert predictors["num_vehicles"].tolist() == [0, 3, 5]
    assert predictors["num_vehicles"].dtype == np.int64
    assert set(UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS) <= set(predictors.columns)
    without = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=frame.table("household").drop(columns=["num_vehicles"]),
        time_period="2024",
        household_weights=frame.weights_for("household").values,
    )
    with pytest.raises(KeyError, match="num_vehicles"):
        recipient_predictors(without, _Engine())


def test_fuel_flag_evidence_records_both_sides_at_design_weights() -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import fuel_flag_evidence

    donor = pd.DataFrame(
        {
            "household_weight": [1.0, 1.0, 2.0],
            "num_vehicles": [0, 1, 2],
            "petrol_spending": [0.0, 0.0, 10.0],
            "diesel_spending": [0.0, 0.0, 0.0],
        }
    )
    recipient = pd.DataFrame(
        {"num_vehicles": [0, 1, 1], "has_fuel_consumption": [False, True, False]}
    )
    evidence = fuel_flag_evidence(
        donor,
        recipient,
        recipient_weights=[1.0, 1.0, 2.0],
        ice_share_receipt={"rate": 0.9},
    )
    assert evidence["ice_share"] == {"rate": 0.9}
    assert evidence["donor"] == {
        "households": 3,
        "with_vehicles_share": 0.75,
        "positive_fuel_share": 0.5,
        "positive_fuel_share_among_vehicle_households": pytest.approx(2 / 3),
    }
    assert evidence["recipient"] == {
        "households": 3,
        "with_vehicles_share": 0.75,
        "flagged_share": 0.25,
    }


def test_post_imputation_rake_fits_all_four_need_margins_in_kwh() -> None:
    # Regression for the licensed-build finding: the manifest declares a
    # four-margin post-imputation rake (income -> tenure -> accommodation ->
    # region); the incumbent hits the tenure/accommodation cells to ~1 %.
    # Since microcosm#890 the margins are the vendored NEED mean kWh (the
    # shape), the rake runs in kWh at the FY2024-25 QEP average prices paid,
    # geography by geography, the gas connection is imposed at the published
    # meter share and one factor per fuel levels the frame to the DESNZ total.
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.energy_pricing import (
        ELECTRICITY_KWH,
        ENGLAND_AND_WALES_GEOGRAPHY_ID,
        GAS_KWH,
        SCOTLAND_GEOGRAPHY_ID,
        kwh_to_spend,
        rake_energy_kwh,
        spend_to_kwh,
    )
    from microcosm.build.uk_runtime.lcfs_consumption import (
        energy_spend_to_kwh,
        lcfs_energy_pricing,
        rake_recipient_energy,
    )

    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    energy = lcfs_energy_pricing(stage)
    assert energy is not None
    assert energy.gas_connected == "published_meter_share"
    assert energy.disconnect_rule == "identity_uniform_order"
    assert energy.disconnect_seed == 0
    rng = np.random.default_rng(3)
    n = 2400
    household = pd.DataFrame(
        {
            "electricity_consumption": rng.uniform(400.0, 2000.0, n),
            "gas_consumption": np.where(
                rng.random(n) < 0.05, 0.0, rng.uniform(300.0, 1500.0, n)
            ),
        }
    )
    income = rng.uniform(5e3, 2e5, n)
    tenure = rng.choice(["OWNED_OUTRIGHT", "RENT_PRIVATELY", "RENT_FROM_COUNCIL"], n)
    accommodation = rng.choice(["HOUSE_DETACHED", "FLAT", "OTHER"], n)
    region = rng.choice(["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"], n)
    weights = rng.uniform(0.5, 2.0, n)

    raked, receipt = rake_recipient_energy(
        household,
        energy=energy,
        region=region,
        income=income,
        tenure=tenure,
        accommodation=accommodation,
        weights=weights,
        iterations=50,
        identity=np.arange(1, n + 1),
    )
    assert receipt["gas_connection"]["disconnect_rule"] == "identity_uniform_order"
    assert receipt["gas_connection"]["seed"] == 0
    elec = raked["electricity_consumption"].to_numpy(dtype=float)
    gas = raked["gas_consumption"].to_numpy(dtype=float)
    factor = receipt["level_factor"]
    margins = energy.margins.targets

    def wmean_kwh(spend, fuel, mask):
        # NEED gas means are per gas-metered household: average gas over the
        # connected (positive) rows of the cell only, electricity over all.
        if fuel == "gas":
            mask = mask & (spend > 0)
        kwh = spend_to_kwh(
            spend[mask],
            frs_region=region[mask],
            fuel=fuel,
            prices=energy.prices,
            connected=None if fuel == "electricity" else spend[mask] > 0,
        )
        return float((kwh * weights[mask]).sum() / weights[mask].sum())

    # Region is the last margin swept, so it fits the levelled target exactly.
    london = region == "LONDON"
    target = (
        margins["region"][(ENGLAND_AND_WALES_GEOGRAPHY_ID, "london")][ELECTRICITY_KWH]
        * factor[ELECTRICITY_KWH]
    )
    assert abs(wmean_kwh(elec, "electricity", london) - target) / target < 1e-6
    scotland = region == "SCOTLAND"
    target = (
        margins["region"][(SCOTLAND_GEOGRAPHY_ID, "all_dwellings")][GAS_KWH]
        * factor[GAS_KWH]
    )
    assert abs(wmean_kwh(gas, "gas", scotland) - target) / target < 1e-6
    assert receipt["fit"]["region"]["max_abs_relative_deviation"][GAS_KWH] < 1e-9
    # Earlier margins settle within a band over 50 iterations (the synthetic
    # categories are mutually inconsistent, so the sweep compromises), per
    # geography: E&W owner-occupiers against the E&W row, not Scotland's.
    ew_owner = np.isin(region, ["LONDON", "WALES"]) & (tenure == "OWNED_OUTRIGHT")
    target = (
        margins["tenure"][(ENGLAND_AND_WALES_GEOGRAPHY_ID, "owner_occupied")][GAS_KWH]
        * factor[GAS_KWH]
    )
    assert abs(wmean_kwh(gas, "gas", ew_owner) - target) / target < 0.15
    # The level: the design-weighted kWh totals are the published DESNZ totals.
    for fuel, spend in ((ELECTRICITY_KWH, elec), (GAS_KWH, gas)):
        kwh = spend_to_kwh(
            spend,
            frs_region=region,
            fuel=fuel.split("_")[0],
            prices=energy.prices,
            connected=None if fuel == ELECTRICITY_KWH else spend > 0,
        )
        assert float(np.dot(kwh, weights)) == pytest.approx(
            energy.level[fuel], rel=1e-6
        )
        assert receipt["level"][fuel]["frame_kwh_after"] == pytest.approx(
            energy.level[fuel], rel=1e-9
        )
    # The gas connection: each GB region sits at its published meter share
    # (the synthetic draw starts at 95 % connected, above every share), and
    # Northern Ireland keeps the positive-gas rule.
    shares = energy.connection_shares
    assert shares is not None
    for name in ("LONDON", "WALES", "SCOTLAND"):
        mask = region == name
        achieved = float(weights[mask & (gas > 0)].sum() / weights[mask].sum())
        assert abs(achieved - shares[name]) < 0.01, name
        entry = receipt["gas_connection"]["by_region"][name]
        assert entry["rows_disconnected"] > 0 and entry["shortfall"] == 0.0
    ni = region == "NORTHERN_IRELAND"
    assert receipt["gas_connection"]["by_region"]["NORTHERN_IRELAND"]["rule"] == (
        "positive_gas_spend"
    )
    zero_gas = household["gas_consumption"].to_numpy() == 0.0
    assert (gas[ni & ~zero_gas] > 0.0).all()
    # With tenure as the last margin swept it fits to the sweep's own tolerance.
    in_kwh = energy_spend_to_kwh(household, energy=energy, region=region)
    two_margin, _ = rake_energy_kwh(
        in_kwh,
        margins=energy.margins,
        frs_region=region,
        income=income,
        weights=weights,
        iterations=50,
        tenure=tenure,
        gas_connected=in_kwh[GAS_KWH].to_numpy(dtype=float) > 0,
    )
    gas_two = two_margin[GAS_KWH].to_numpy(dtype=float)
    connected_owner = ew_owner & (gas_two > 0)
    fitted = float(
        (gas_two[connected_owner] * weights[connected_owner]).sum()
        / weights[connected_owner].sum()
    )
    shape = margins["tenure"][(ENGLAND_AND_WALES_GEOGRAPHY_ID, "owner_occupied")][
        GAS_KWH
    ]
    assert abs(fitted - shape) / shape < 1e-6
    # Northern Ireland has no NEED table: untouched by every margin, so its
    # electricity moves by the level factor alone through the NI pricing.
    before_kwh = spend_to_kwh(
        household["electricity_consumption"].to_numpy()[ni],
        frs_region=region[ni],
        fuel="electricity",
        prices=energy.prices,
    )
    np.testing.assert_allclose(
        elec[ni],
        kwh_to_spend(
            before_kwh * factor[ELECTRICITY_KWH],
            frs_region=region[ni],
            fuel="electricity",
            prices=energy.prices,
        ),
    )
    # Disconnected and never-connected households carry no gas fixed cost.
    assert (gas[zero_gas] == 0.0).all()
    assert receipt["unit"] == "kwh"
    assert receipt["margins"] == ["income", "tenure", "accommodation", "region"]
    assert receipt["margins_period_value"] == 2024
    assert 0.7 < receipt["gas_connected_share"] < 0.9
    assert receipt["gas_rake_population"] == "gas_connected_rows"
    assert receipt["gas_connected_rows"] == int((gas > 0).sum())


def _synthetic_lcfs_donor(
    n: int = 240, seed: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = np.arange(n)
    household = {
        "case": rows + 1,
        "g018": 1 + rows % 3,
        "g019": rows % 3,
        "gorx": 1 + rows % 12,
        "a124": rows % 4,
        "p389p": rng.uniform(100.0, 1500.0, n),
        "p344p": rng.uniform(150.0, 2000.0, n),
        "weighta": rng.uniform(0.5, 2.0, n),
        "a122": 1 + rows % 8,
        "a121": 1 + rows % 8,
        "b226": rng.uniform(0.0, 30.0, n),
        "b489": rng.uniform(0.0, 30.0, n),
        "b490": rng.uniform(0.0, 20.0, n),
        "p537": rng.uniform(10.0, 60.0, n),
    }
    for code in (
        "p601",
        "p602",
        "p603",
        "p604",
        "p605",
        "p606",
        "p607",
        "p608",
        "p609",
        "p610",
        "p611",
        "p612",
    ):
        household[code] = rng.uniform(1.0, 200.0, n)
    # No fuel without a car, and half the one-car diaries record no purchase
    # either (the two-week diary's zeros the road-fuel redraw corrects).
    no_fuel = (rows % 4 == 0) | (rows % 8 == 1)
    household["c72211"] = np.where(no_fuel, 0.0, rng.uniform(5.0, 80.0, n))
    household["c72212"] = np.where(no_fuel, 0.0, rng.uniform(0.0, 40.0, n))
    # Other motor fuels inside 07.2.2, deterministic so the draws below keep
    # their stream.
    household["c72213"] = np.where(rows % 10 == 1, 0.5, 0.0)
    for code in BUS_FARE_LCFS_CODES:
        household[code] = np.where(rng.random(n) < 0.55, 0.0, rng.uniform(1.0, 30.0, n))
    person = pd.DataFrame(
        {
            "case": rows + 1,
            "b303p": rng.uniform(0.0, 900.0, n),
            "b3262p": rng.uniform(0.0, 100.0, n),
            "p049p": rng.uniform(0.0, 200.0, n),
        }
    )
    return person, pd.DataFrame(household)


def test_stage_transform_prices_bus_fares_from_journeys() -> None:
    """End to end on synthetic inputs: the committed declaration minus the engine.

    The uprating step needs the installed engine and is dropped; the QRF is
    shrunk to four trees. Everything else is the committed lcfs_consumption
    stage (#890 A, E2; #930 pricing).
    """

    import dataclasses
    from types import SimpleNamespace

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.source_manifest import SourceOperationSpec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS,
        UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS,
    )

    committed = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    operations = []
    for operation in committed.operations:
        if operation.kind == "uprate_donor_columns":
            continue
        if operation.kind == "fit_weighted_qrf_chain":
            operation = SourceOperationSpec(
                kind=operation.kind,
                parameters={**operation.parameters, "n_estimators": 4},
            )
        operations.append(operation)
    stage = dataclasses.replace(committed, operations=tuple(operations))

    rng = np.random.default_rng(21)
    n = 360
    regions = np.array(
        ["LONDON", "SOUTH_EAST", "NORTH_WEST", "SCOTLAND", "WALES", "NORTHERN_IRELAND"]
    )[np.arange(n) % 6]
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1),
            "household_weight": rng.uniform(0.5, 3.0, n),
            "region": regions,
            "tenure_type": np.array(
                ["OWNED_OUTRIGHT", "RENT_PRIVATELY", "RENT_FROM_COUNCIL"]
            )[np.arange(n) % 3],
            "accommodation_type": np.array(
                ["HOUSE_DETACHED", "FLAT", "HOUSE_TERRACED"]
            )[np.arange(n) % 3],
            "num_vehicles": np.arange(n) % 3,
            "household_gross_income": rng.uniform(5e3, 9e4, n),
        }
    )
    ages = rng.integers(1, 90, 2 * n).astype(float)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 2 * n + 1),
            "person_benunit_id": np.repeat(np.arange(1, n + 1), 2),
            "person_household_id": np.repeat(np.arange(1, n + 1), 2),
            "age": ages,
            # The nts_bus_travel stage's cells: journeys per series and the
            # statutory concessionary eligibility (here, 66 and over).
            "bus_in_london_trips": np.where(
                np.repeat(regions, 2) == "LONDON", rng.uniform(0, 200, 2 * n), 0.0
            ),
            "other_local_bus_trips": np.where(
                np.repeat(regions, 2) == "LONDON", 0.0, rng.uniform(0, 120, 2 * n)
            ),
            "bus_pass_eligible": ages >= 66,
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": np.arange(1, n + 1)}),
        household=household,
        time_period="2024",
    )

    class _Engine:
        country = "uk"

        def variable_metadata(self, name):
            return SimpleNamespace(entity="household")

        def materialize(self, frame, variables, period):
            rows = len(frame.table("household"))
            values = {
                "employment_income": rng.uniform(0.0, 5e4, rows),
                "self_employment_income": rng.uniform(0.0, 5e3, rows),
                "private_pension_income": rng.uniform(0.0, 1e4, rows),
                "hbai_household_net_income": frame.table("household")[
                    "household_gross_income"
                ].to_numpy()
                * 0.8,
            }
            return {name: values[name] for name in variables}

    donor_person, donor_household = _synthetic_lcfs_donor()
    transform = UKLCFSConsumptionStageTransform(
        stage=stage,
        engine=_Engine(),
        lcfs_household=donor_household,
        lcfs_person=donor_person,
    )
    result = transform(frame)

    out = result.table("household")
    assert set(UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS) <= set(out.columns)
    # The 18 chain targets, then the road-fuel redraw's total and petrol share.
    assert len(transform.fit_weight_records) == 20
    assert transform.fit_weight_records[-1].fit_name.endswith(
        ":petrol_share_of_road_fuel"
    )
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_CONSUMPTION_MASS_CONSERVATION_REASON,
    )

    receipt = result.mass_log[-1]
    assert receipt.reason == UK_LCFS_CONSUMPTION_MASS_CONSERVATION_REASON
    assert receipt.old_total == receipt.new_total > 0
    assert receipt.declared_factor == 1.0
    assert set(UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS) <= set(
        transform.stage.operations[4].parameters["predictors"]
    )
    evidence = transform.checkpoint_metadata()["evidence"]
    assert {
        "support_clip",
        "has_fuel_consumption",
        "bus_pricing",
        "energy_pricing",
        "energy_rake",
    } <= set(evidence)
    assert evidence["energy_pricing"]["payment_method"] == "all"
    assert evidence["energy_pricing"]["vat_treatment"] == "including_vat"
    assert evidence["energy_pricing"]["gas_connection"]["rule"] == (
        "published_meter_share"
    )
    assert evidence["energy_rake"]["unit"] == "kwh"
    assert evidence["energy_rake"]["margins_period_value"] == 2024
    assert evidence["energy_rake"]["level"]["gas_kwh"]["factor"] > 0
    assert evidence["energy_rake"]["gas_connection"]["rule"] == "published_meter_share"
    # The uprating step is dropped in this engine-free run, so no litres audit.
    assert "fuel_litres_audit" not in evidence
    assert "donor_uprating" not in evidence
    # Fuel: no vehicles means no fuel; the fuel-buyer share came from VEH1103.
    no_vehicle = household["num_vehicles"].to_numpy() == 0
    assert (out.loc[no_vehicle, "petrol_spending"] == 0.0).all()
    assert 0.95 < evidence["has_fuel_consumption"]["ice_share"]["rate"] < 0.97
    # Road fuel: petrol plus diesel at prior weights is ONS 07.2.2 for 2024
    # less the donor's own other-fuels share (#1113).
    level = evidence["road_fuel_level"]
    assert level["coicop"] == "07.2.2" and level["period_value"] == 2024
    weight = donor_household["weighta"].to_numpy()
    fuels = donor_household[["c72211", "c72212", "c72213"]].to_numpy()
    assert level["other_fuels_share"] == pytest.approx(
        weight @ fuels[:, 2] / (weight @ fuels.sum(axis=1))
    )
    assert level["level"] == pytest.approx(
        level["published"] * (1.0 - level["other_fuels_share"])
    )
    road_fuel = out[["petrol_spending", "diesel_spending"]].to_numpy().sum(axis=1)
    assert result.weights_for("household").values @ road_fuel == pytest.approx(
        level["level"]
    )
    assert level["frame_after"] == pytest.approx(level["level"])
    assert (out.loc[no_vehicle, "diesel_spending"] == 0.0).all()
    # Incidence: every flagged fuel-car household carries road fuel (#1113).
    flagged = out["has_fuel_consumption"].to_numpy(dtype=bool)
    assert (road_fuel[flagged] > 0).all() and (road_fuel[~flagged] == 0).all()
    incidence = evidence["road_fuel_incidence"]
    assert incidence["flagged_zero_before"] > 0
    assert incidence["flagged_zero_after"] == 0
    assert incidence["regimes"]["road_fuel_total"] == "positive_only"
    assert level["frame_before"] == pytest.approx(incidence["weighted_total_after"])
    # The receipts the stage writes are the ones its gates read.
    from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

    # Totals contain their levelled components (#1113).
    energy = out[["electricity_consumption", "gas_consumption"]].sum(axis=1)
    assert (out["housing_water_and_electricity_consumption"] >= energy).all()
    assert (out["transport_consumption"] >= road_fuel).all()
    recomposed = evidence["recomposed_totals"]["parents"]
    assert recomposed["transport_consumption"]["weighted_components"] == (
        pytest.approx(level["level"])
    )
    assert recomposed["housing_water_and_electricity_consumption"][
        "weighted_total"
    ] == pytest.approx(
        result.weights_for("household").values
        @ out["housing_water_and_electricity_consumption"].to_numpy()
    )
    gates = [
        gate
        for gate in load_country_spec("uk").gates.gates
        if gate.id
        in {
            "uk_stage_lcfs_consumption_road_fuel_incidence",
            "uk_stage_lcfs_consumption_road_fuel_level",
            "uk_stage_lcfs_consumption_recomposed_totals",
        }
    ]
    assert len(gates) == 3
    for gate in gates:
        checked = uk_stage_health_gate(
            evidence=evidence,
            stage="lcfs_consumption",
            check=gate.parameters["check"],
            parameters=gate.parameters,
        )
        assert checked.passed, checked.failures
    # Bus: fares are journeys times the published yield in every priced region;
    # Wales keeps the chain's raw draw, clipped to donor support.
    pricing = evidence["bus_pricing"]
    assert pricing["chain_conditioned_on"] == "raw_draw"
    assert pricing["unpriced_regions"] == ["WALES"]
    wales = regions == "WALES"
    assert pricing["households_priced"] == int((~wales).sum())
    assert set(pricing["by_area"]) == {
        "london_series",
        "england_outside_london",
        "scotland",
        "northern_ireland",
    }
    fares = out["bus_fare_spending"].to_numpy()
    prices = pricing["prices"]
    fare_per_trip = {
        "LONDON": prices["england_outside_london"]["fare_per_trip"],
        "SOUTH_EAST": prices["england_outside_london"]["fare_per_trip"],
        "NORTH_WEST": prices["england_outside_london"]["fare_per_trip"],
        "SCOTLAND": prices["scotland"]["fare_per_trip"],
        "NORTHERN_IRELAND": prices["northern_ireland"]["fare_per_trip"],
    }
    london_fare = prices["london_series"]["fare_per_trip"]
    expected = np.zeros(n)
    for row in person.itertuples(index=False):
        h = int(row.person_household_id) - 1
        region = regions[h]
        if region == "WALES" or row.bus_pass_eligible:
            continue
        expected[h] += row.bus_in_london_trips * london_fare
        expected[h] += row.other_local_bus_trips * fare_per_trip[region]
    assert np.allclose(fares[~wales], expected[~wales])
    donor_max = donor_household[list(BUS_FARE_LCFS_CODES)].sum(axis=1).max() * (
        365.25 / 7
    )
    assert fares[wales].max() <= donor_max + 1e-6
    # Every published factor is on the receipt beside its source records.
    london = prices["london_series"]
    assert london["yield_per_fare_paying_boarding"] == pytest.approx(
        london["receipts"]["value"]
        / (london["boardings"]["value"] - london["concessionary"]["value"])
    )
    assert london["boardings_per_trip"] == pytest.approx(
        london["boardings"]["value"]
        / (london["trips_per_person"]["value"] * london["population"]["value"])
    )
    assert pricing["by_area"]["london_series"]["frame_implied_boardings"] > 0


def test_fuel_litres_audit_reads_the_vendored_prices_litres_and_obr_split() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import fuel_litres_audit
    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    draws = pd.DataFrame(
        {"petrol_spending": [1000.0, 0.0, 500.0], "diesel_spending": [0.0, 300.0, 0.0]}
    )
    weights = np.array([2.0, 1.0, 1.0])

    audit = fuel_litres_audit(draws, weights=weights, stage=stage)

    assert audit is not None
    assert audit["period_value"] == 2024 and audit["fiscal_start"] == "2024-04-01"
    price = float(
        vendored_rows(
            "road_fuel_anchors.json",
            concept="desnz.road_fuel.annual_ulsp_pump_price",
            period_type="calendar_year",
            period_value=2024,
        )[0]["value"]
    )
    petrol = audit["fuels"]["petrol_spending"]
    assert petrol["weighted_spend_gbp"] == pytest.approx(2500.0)
    assert petrol["frame_litres"] == pytest.approx(2500.0 / (price / 100.0))
    hmrc = float(
        vendored_rows(
            "road_fuel_anchors.json",
            concept="hmrc.hydrocarbon_oils.total_petrol_quantity",
            fiscal_start="2024-04-01",
        )[0]["value"]
    )
    assert petrol["hmrc_litres_all_road_users"] == hmrc
    # OBR FY2024-25: cars GBP 14.4bn of GBP 24.7bn fuel duty.
    assert audit["obr_fuel_duty_receipts"]["cars_share"] == pytest.approx(
        14.4 / 24.7, abs=1e-3
    )
    assert petrol["cars_litres_benchmark"] == pytest.approx(
        hmrc * audit["obr_fuel_duty_receipts"]["cars_share"]
    )
    assert petrol["frame_over_cars_benchmark"] == pytest.approx(
        petrol["frame_litres"] / petrol["cars_litres_benchmark"]
    )
    assert set(audit["fuels"]) == {"petrol_spending", "diesel_spending"}
    assert audit["gated"] is False
    # Only the total compares like with like (#1113).
    assert audit["per_fuel_ratio_basis"] == (
        "uniform cars share, not a per-fuel benchmark"
    )

    class Stage:
        operations = ()

    assert fuel_litres_audit(draws, weights=weights, stage=Stage()) is None


def test_road_fuel_level_reads_the_vendored_ons_row() -> None:
    """The declared level is ONS 07.2.2 for 2024 less the other-fuels share (#1113)."""

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        road_fuel_level,
        road_fuel_level_operation,
    )
    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    declared = road_fuel_level_operation(stage)
    assert declared is not None
    kinds = [operation.kind for operation in stage.operations]
    assert kinds.index("redraw_zero_road_fuel") == kinds.index("zero_when_false") + 1
    assert kinds.index("level_road_fuel") == kinds.index("redraw_zero_road_fuel") + 1

    level = road_fuel_level(declared, other_fuels_share=0.004)

    (row,) = vendored_rows(
        "ons_household_expenditure_facts.json",
        concept="ons.household_expenditure.personal_transport_fuels_lubricants",
        period_type="calendar_year",
        period_value=2024,
        geography_id="K02000001",
        dimensions={"coicop": "07.2.2", "frequency": "annual"},
    )
    assert level.published == float(row["value"])
    assert level.level == pytest.approx(float(row["value"]) * 0.996)
    assert level.receipt["source_record_id"] == row["source_record_id"]
    assert level.receipt["columns"] == ["petrol_spending", "diesel_spending"]
    assert level.receipt["other_fuels_codes"] == ["c72213"]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"columns": ["petrol_spending"]}, "must level"),
        ({"resource": "road_fuel_anchors.json"}, "must read"),
        ({"other_fuels_rule": "none"}, "unsupported"),
        ({"other_fuels_codes": ["c72213", "c72214"]}, "other_fuels_codes"),
        ({"period_type": "fiscal_year"}, "calendar-year"),
        ({"period_value": 2019}, "expected one"),
    ],
)
def test_road_fuel_level_refuses_a_wrong_declaration(change, match) -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        road_fuel_level,
        road_fuel_level_operation,
    )

    declared = road_fuel_level_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    with pytest.raises(ValueError, match=match):
        road_fuel_level({**declared, **change}, other_fuels_share=0.0)


def test_road_fuel_level_refuses_a_share_outside_the_unit_interval() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        road_fuel_level,
        road_fuel_level_operation,
    )

    declared = road_fuel_level_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    for share in (-0.01, 1.0):
        with pytest.raises(ValueError, match="outside"):
            road_fuel_level(declared, other_fuels_share=share)


def test_level_road_fuel_scales_both_fuels_by_one_factor() -> None:
    """One factor: the prior-weighted total is the level and the mix is kept."""

    from microcosm.build.uk_runtime.lcfs_consumption import (
        RoadFuelLevel,
        level_road_fuel,
    )

    draws = pd.DataFrame(
        {
            "petrol_spending": [1000.0, 0.0, 500.0, 0.0],
            "diesel_spending": [0.0, 300.0, 250.0, 0.0],
            "food_consumption": [1.0, 2.0, 3.0, 4.0],
        }
    )
    weights = np.array([2.0, 1.0, 1.0, 5.0])
    level = RoadFuelLevel(5000.0, 0.02, {"operation": "level_road_fuel"})

    levelled, receipt = level_road_fuel(draws, level=level, weights=weights)

    before = 2.0 * 1000.0 + 300.0 + 750.0
    assert receipt["frame_before"] == pytest.approx(before)
    assert receipt["factor"] == pytest.approx(4900.0 / before)
    assert receipt["frame_after"] == pytest.approx(4900.0)
    np.testing.assert_allclose(
        levelled[["petrol_spending", "diesel_spending"]].to_numpy(),
        draws[["petrol_spending", "diesel_spending"]].to_numpy() * receipt["factor"],
    )
    assert receipt["petrol_share_of_level"] == pytest.approx(2500.0 / before)
    assert levelled["food_consumption"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert levelled.loc[3, ["petrol_spending", "diesel_spending"]].tolist() == [
        0.0,
        0.0,
    ]
    with pytest.raises(ValueError, match="no road-fuel spend"):
        level_road_fuel(draws * 0.0, level=level, weights=weights)


def test_other_road_fuel_share_is_the_donor_weighted_c72213_share() -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import (
        lcfs_other_road_fuel_share,
    )

    donor = pd.DataFrame(
        {
            "WEIGHTA": [1.0, 3.0],
            "C72211": [10.0, 0.0],
            "C72212": [0.0, 5.0],
            "C72213": [1.0, 0.0],
        }
    )
    assert lcfs_other_road_fuel_share(donor) == pytest.approx(1.0 / 26.0)
    with pytest.raises(ValueError, match="no COICOP 07.2.2"):
        lcfs_other_road_fuel_share(donor.assign(C72211=0.0, C72212=0.0, C72213=0.0))
    with pytest.raises(ValueError, match="c72213"):
        lcfs_other_road_fuel_share(donor.drop(columns="C72213"))


def test_road_fuel_level_is_a_no_op_when_undeclared() -> None:
    import dataclasses

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import lcfs_road_fuel_level

    committed = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    stage = dataclasses.replace(
        committed,
        operations=tuple(
            operation
            for operation in committed.operations
            if operation.kind != "level_road_fuel"
        ),
    )
    draws = pd.DataFrame({"petrol_spending": [1.0], "diesel_spending": [2.0]})

    levelled, receipt = lcfs_road_fuel_level(
        stage, draws, weights=[1.0], lcfs_household=pd.DataFrame()
    )

    assert receipt is None and levelled is draws


_ROAD_FUEL = ["petrol_spending", "diesel_spending"]


def _road_fuel_sides(
    seed: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A donor with none, petrol, diesel and both, and zeroed recipient draws."""

    rng = np.random.default_rng(seed)

    def side(n: int) -> pd.DataFrame:
        rows = np.arange(n)
        return pd.DataFrame(
            {
                "household_adult_count": 1 + rows % 3,
                "household_child_count": rows % 2,
                "region": np.array(["LONDON", "WALES", "SCOTLAND"])[rows % 3],
                "employment_income": rng.uniform(0.0, 5e4, n),
                "self_employment_income": rng.uniform(0.0, 5e3, n),
                "private_pension_income": rng.uniform(0.0, 1e4, n),
                "hbai_household_net_income": rng.uniform(1e4, 6e4, n),
                "tenure_type": np.array(["OWNED_OUTRIGHT", "RENT_PRIVATELY"])[rows % 2],
                "accommodation_type": np.array(["FLAT", "HOUSE_DETACHED"])[rows % 2],
                "num_vehicles": rows % 3,
            }
        )

    donor = side(200)
    kind = np.arange(200) % 4
    donor["petrol_spending"] = np.where(kind % 2 == 1, rng.uniform(300, 2e3, 200), 0.0)
    donor["diesel_spending"] = np.where(kind >= 2, rng.uniform(300, 2e3, 200), 0.0)
    donor["household_weight"] = rng.uniform(500.0, 2000.0, 200)
    recipient = side(40)
    recipient["has_fuel_consumption"] = recipient["num_vehicles"].to_numpy() > 0
    rows = np.arange(40)
    draws = pd.DataFrame(
        {
            "petrol_spending": np.where(rows % 4 == 1, 800.0, 0.0),
            "diesel_spending": np.where(rows % 4 == 2, 600.0, 0.0),
            "food_and_non_alcoholic_beverages_consumption": 1.0 + rows,
        }
    )
    draws.loc[~recipient["has_fuel_consumption"], _ROAD_FUEL] = 0.0
    return donor, recipient, draws


def _redraw_declaration() -> dict:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        road_fuel_incidence_operation,
    )

    declared = road_fuel_incidence_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    assert declared is not None
    return {**declared, "n_estimators": 10}


def test_redraw_gives_every_flagged_household_road_fuel() -> None:
    """Zero draws among flagged households come from the positive-fuel donors (#1113)."""

    from microcosm.build.uk_runtime.lcfs_consumption import redraw_zero_road_fuel

    donor, recipient, draws = _road_fuel_sides()
    flagged = recipient["has_fuel_consumption"].to_numpy(dtype=bool)
    weights = np.linspace(1.0, 2.0, 40)

    result = redraw_zero_road_fuel(
        draws,
        donor=donor,
        recipient=recipient,
        flagged=flagged,
        household_ids=np.arange(101, 141),
        weights=weights,
        parameters=_redraw_declaration(),
    )

    before = draws[_ROAD_FUEL].to_numpy()
    after = result.draws[_ROAD_FUEL].to_numpy()
    zero = flagged & (before.sum(axis=1) == 0)
    assert zero.sum() > 0
    assert (after.sum(axis=1)[flagged] > 0).all()
    assert (after.sum(axis=1)[~flagged] == 0).all() and (after >= 0).all()
    np.testing.assert_array_equal(after[~zero], before[~zero])
    donor_total = donor[_ROAD_FUEL].to_numpy().sum(axis=1)
    assert after.sum(axis=1)[zero].max() <= donor_total.max() + 1e-9
    assert result.draws["food_and_non_alcoholic_beverages_consumption"].equals(
        draws["food_and_non_alcoholic_beverages_consumption"]
    )
    receipt = result.receipt
    assert receipt["flagged_zero_before"] == int(zero.sum())
    assert receipt["flagged_zero_after"] == 0 and receipt["unflagged_with_fuel"] == 0
    assert receipt["flagged_zero_share_before"] == pytest.approx(
        weights[zero].sum() / weights[flagged].sum()
    )
    assert receipt["training_rows"] == int((donor_total > 0).sum())
    assert receipt["regimes"] == {
        "road_fuel_total": "positive_only",
        "petrol_share_of_road_fuel": "zero_inflated_positive",
    }
    assert receipt["mean_positive_before"] == pytest.approx(
        weights[flagged & ~zero]
        @ before.sum(axis=1)[flagged & ~zero]
        / weights[flagged & ~zero].sum()
    )
    assert receipt["weighted_total_after"] == pytest.approx(weights @ after.sum(axis=1))
    assert [record.fit_name for record in result.fit_weight_records] == [
        "uk_lcfs_2023_24_consumption:road_fuel_total",
        "uk_lcfs_2023_24_consumption:petrol_share_of_road_fuel",
    ]


def test_redraw_is_keyed_on_household_identity_not_row_order() -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import redraw_zero_road_fuel

    donor, recipient, draws = _road_fuel_sides()
    flagged = recipient["has_fuel_consumption"].to_numpy(dtype=bool)
    ids = np.arange(101, 141)
    weights = np.linspace(1.0, 2.0, 40)
    parameters = _redraw_declaration()
    base = redraw_zero_road_fuel(
        draws,
        donor=donor,
        recipient=recipient,
        flagged=flagged,
        household_ids=ids,
        weights=weights,
        parameters=parameters,
    )
    order = np.random.default_rng(0).permutation(40)

    shuffled = redraw_zero_road_fuel(
        draws.iloc[order].reset_index(drop=True),
        donor=donor,
        recipient=recipient.iloc[order].reset_index(drop=True),
        flagged=flagged[order],
        household_ids=ids[order],
        weights=weights[order],
        parameters=parameters,
    )

    np.testing.assert_allclose(
        shuffled.draws[_ROAD_FUEL].to_numpy(),
        base.draws[_ROAD_FUEL].to_numpy()[order],
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"columns": ["petrol_spending"]}, "must redraw"),
        ({"flag": "num_vehicles"}, "has_fuel_consumption"),
        ({"rule": "zero_inflated"}, "unsupported"),
        ({"seed": "0"}, "integer seed"),
        ({"salt": ""}, "salt"),
    ],
)
def test_redraw_refuses_a_wrong_declaration(change, match) -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import redraw_zero_road_fuel

    donor, recipient, draws = _road_fuel_sides()
    with pytest.raises(ValueError, match=match):
        redraw_zero_road_fuel(
            draws,
            donor=donor,
            recipient=recipient,
            flagged=recipient["has_fuel_consumption"].to_numpy(dtype=bool),
            household_ids=np.arange(40),
            weights=np.ones(40),
            parameters={**_redraw_declaration(), **change},
        )


def test_redraw_is_a_no_op_when_undeclared() -> None:
    import dataclasses

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import lcfs_road_fuel_incidence

    committed = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    stage = dataclasses.replace(
        committed,
        operations=tuple(
            operation
            for operation in committed.operations
            if operation.kind != "redraw_zero_road_fuel"
        ),
    )
    donor, recipient, draws = _road_fuel_sides()

    result = lcfs_road_fuel_incidence(
        stage,
        draws,
        donor=donor,
        recipient=recipient,
        household_ids=np.arange(40),
        weights=np.ones(40),
    )

    assert result.draws is draws and result.receipt is None
    assert result.fit_weight_records == ()


def _recompose_declaration() -> dict:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        recompose_from_remainder_operation,
    )

    declared = recompose_from_remainder_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    assert declared is not None
    return declared


def test_recompose_writes_each_total_around_its_levelled_parts() -> None:
    """Totals keep the chain's own split and hold the levelled parts (#1113)."""

    from microcosm.build.uk_runtime.lcfs_consumption import recompose_parent_totals
    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    # The chain's draws of the parts, before any step re-levelled them.
    drawn = pd.DataFrame(
        {
            "domestic_energy_consumption": [2000.0, 900.0, 0.0],
            "petrol_spending": [800.0, 0.0, 0.0],
            "diesel_spending": [0.0, 700.0, 0.0],
        }
    )
    # The levelled parts, beside the totals as the chain drew them.
    draws = pd.DataFrame(
        {
            "housing_water_and_electricity_consumption": [7000.0, 600.0, 800.0],
            "electricity_consumption": [900.0, 600.0, 0.0],
            "gas_consumption": [700.0, 0.0, 0.0],
            "transport_consumption": [2000.0, 1000.0, 0.0],
            "petrol_spending": [1500.0, 0.0, 0.0],
            "diesel_spending": [0.0, 900.0, 0.0],
            "food_and_non_alcoholic_beverages_consumption": [1.0, 2.0, 3.0],
        }
    )
    weights = np.array([2.0, 1.0, 3.0])

    out, receipt = recompose_parent_totals(
        draws, drawn=drawn, parameters=_recompose_declaration(), weights=weights
    )

    # Housing: 7000 - 2000 + 1600; 600 - 900 floors at 0, + 600; 800 + 0.
    assert out["housing_water_and_electricity_consumption"].tolist() == [
        6600.0,
        600.0,
        800.0,
    ]
    # Transport: 2000 - 800 + 1500; 1000 - 700 + 900; 0.
    assert out["transport_consumption"].tolist() == [2700.0, 1200.0, 0.0]
    assert out["food_and_non_alcoholic_beverages_consumption"].tolist() == [
        1.0,
        2.0,
        3.0,
    ]
    housing = receipt["parents"]["housing_water_and_electricity_consumption"]
    assert housing["drawn_subtracts"] == ["domestic_energy_consumption"]
    assert housing["weighted_drawn_total"] == 2.0 * 7000.0 + 600.0 + 3.0 * 800.0
    assert housing["rows_floored"] == 1
    assert housing["weighted_floored_mass"] == 300.0
    assert housing["weighted_remainder"] == pytest.approx(
        housing["weighted_drawn_total"]
        - housing["weighted_drawn_subtracts"]
        + housing["weighted_floored_mass"]
    )
    assert housing["rows_below_components"] == 0
    transport = receipt["parents"]["transport_consumption"]
    assert transport["weighted_total"] == 2.0 * 2700.0 + 1200.0
    assert transport["weighted_components"] == 2.0 * 1500.0 + 900.0
    uncarried = receipt["uncarried"]
    expected = [
        float(
            vendored_rows(
                "ons_household_expenditure_facts.json",
                concept=concept,
                period_type="calendar_year",
                period_value=2024,
                geography_id="K02000001",
                dimensions={"coicop": coicop, "frequency": "annual"},
            )[0]["value"]
        )
        for coicop, concept in (
            ("04.5.3", "ons.household_expenditure.liquid_fuels"),
            ("04.5.4", "ons.household_expenditure.solid_fuels"),
        )
    ]
    assert [entry["value"] for entry in uncarried["classes"]] == expected
    assert uncarried["total"] == pytest.approx(sum(expected))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda d: d["parents"]["transport_consumption"].update(
                drawn_subtracts=["petrol_spending"]
            ),
            "not its draw less",
        ),
        (
            lambda d: d["parents"]["housing_water_and_electricity_consumption"].update(
                components=["electricity_consumption"]
            ),
            "not its draw less",
        ),
        (lambda d: d["parents"].pop("transport_consumption"), "must declare"),
        (
            lambda d: d["uncarried"].update(resource="family_resources.json"),
            "not a vendored",
        ),
        (
            lambda d: d["uncarried"]["classes"][0].update(concept="ons.other"),
            "expected one",
        ),
    ],
)
def test_recompose_refuses_a_declaration_it_does_not_apply(change, match) -> None:
    import copy

    from microcosm.build.uk_runtime.lcfs_consumption import recompose_parent_totals

    declared = copy.deepcopy(_recompose_declaration())
    change(declared)
    columns = (
        "housing_water_and_electricity_consumption",
        "domestic_energy_consumption",
        "electricity_consumption",
        "gas_consumption",
        "transport_consumption",
        "petrol_spending",
        "diesel_spending",
    )
    draws = pd.DataFrame({column: [1.0] for column in columns})
    with pytest.raises(ValueError, match=match):
        recompose_parent_totals(draws, drawn=draws, parameters=declared, weights=[1.0])


def test_recompose_is_a_no_op_when_undeclared() -> None:
    import dataclasses

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        lcfs_recompose_from_remainder,
    )

    committed = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    stage = dataclasses.replace(
        committed,
        operations=tuple(
            operation
            for operation in committed.operations
            if operation.kind != "recompose_from_remainder"
        ),
    )
    draws = pd.DataFrame({"transport_consumption": [1.0]})

    out, receipt = lcfs_recompose_from_remainder(
        stage, draws, drawn=draws, weights=[1.0]
    )

    assert out is draws and receipt is None


def test_recipient_counts_adults_and_children_by_age_not_engine_flags() -> None:
    """The LCFS G018/G019 split is age 18, which the recipient counts from age.

    The engine's is_adult and is_child flags are deprecated (policyengine-uk
    #1896), so the household counts never ask the engine for them
    (uk-data#486, microcosm#1095).
    """

    from types import SimpleNamespace

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "person_benunit_id": [1, 1, 1, 2, 2],
            "person_household_id": [1, 1, 1, 2, 2],
            "age": [40.0, 10.0, 70.0, 17.0, 18.0],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "num_vehicles": [1, 0],
            "household_gross_income": [3e4, 2e4],
            "household_weight": [1.0, 1.0],
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=household,
        time_period="2024",
    )
    requested: list[str] = []

    class _Engine:
        def variable_metadata(self, name):
            return SimpleNamespace(entity="household")

        def materialize(self, frame, variables, period):
            requested.extend(variables)
            return {name: np.zeros(2) for name in variables}

    result = recipient_predictors(frame, _Engine())
    assert result["household_adult_count"].tolist() == [2.0, 1.0]
    assert result["household_child_count"].tolist() == [1.0, 1.0]
    assert not {"is_adult", "is_child"} & set(requested)
