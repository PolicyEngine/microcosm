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
    )
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
    household["c72211"] = np.where(rows % 4 == 0, 0.0, rng.uniform(5.0, 80.0, n))
    household["c72212"] = np.where(rows % 4 == 0, 0.0, rng.uniform(0.0, 40.0, n))
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


def test_stage_transform_imposes_bus_incidence_and_rakes_fares_to_the_facts() -> None:
    """End to end on synthetic inputs: the committed declaration minus the engine.

    The uprating step needs the installed engine and is dropped; the QRF is
    shrunk to four trees. Everything else is the committed lcfs_consumption
    stage (#890 A, E2, I and D).
    """

    import dataclasses
    from types import SimpleNamespace

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.source_manifest import SourceOperationSpec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS,
        UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS,
    )
    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

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
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 2 * n + 1),
            "person_benunit_id": np.repeat(np.arange(1, n + 1), 2),
            "person_household_id": np.repeat(np.arange(1, n + 1), 2),
            "age": rng.integers(1, 90, 2 * n).astype(float),
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
                "is_adult": np.full(rows, 2.0),
                "is_child": np.zeros(rows),
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
    assert len(transform.fit_weight_records) == 18
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
        "bus_use_incidence",
        "bus_fare_rake",
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
    # Bus: non-user households are zero, user cells hit the published totals.
    incidence = evidence["bus_use_incidence"]
    assert incidence["users_filled_from_positive_regime"] >= 0
    assert (
        incidence["users_in_scope"]
        == incidence["users_drawn_positive"]
        + (incidence["users_filled_from_positive_regime"])
    )
    assert 0.3 < incidence["household_user_share"] < 0.95
    rake = evidence["bus_fare_rake"]
    assert rake["scope"] == "users_only"
    fares = out["bus_fare_spending"].to_numpy()
    weights = result.weights_for("household").values
    london_total = float(
        np.dot(fares[regions == "LONDON"], weights[regions == "LONDON"])
    )
    published_london = float(
        vendored_rows(
            "dft_bus_value_anchors.json",
            concept="dft.local_bus_passenger_fare_receipts",
            fiscal_start="2024-04-01",
            geography_id="E12000007",
        )[0]["value"]
    )
    assert london_total == pytest.approx(published_london)
    ni = regions == "NORTHERN_IRELAND"
    assert float(np.dot(fares[ni], weights[ni])) == pytest.approx(
        49_584_434.28 + 100_498_383.21
    )
    wales = regions == "WALES"
    # The incidence override is imposed only where a fare cell levels the
    # amounts; Wales (no published receipts) keeps the chain's raw draw, so
    # its positive share is the QRF's, not the NTS user share.
    assert incidence["households_outside_scope_keep_raw_draw"] == int(wales.sum())
    assert incidence["users_in_scope"] == int((fares[~wales] > 0).sum())
    donor_max = donor_household[list(BUS_FARE_LCFS_CODES)].sum(axis=1).max() * (
        365.25 / 7
    )
    assert fares[wales].max() <= donor_max + 1e-6
    assert {fit["label"] for fit in rake["fits"]} == {
        "london",
        "england_outside_london",
        "scotland",
        "northern_ireland",
    }


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

    class Stage:
        operations = ()

    assert fuel_litres_audit(draws, weights=weights, stage=Stage()) is None
