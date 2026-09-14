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


def test_post_imputation_rake_fits_all_four_need_margins() -> None:
    # Regression for the licensed-build finding: the manifest declares a
    # four-margin post-imputation rake (income -> tenure -> accommodation ->
    # region), but only the income margin was implemented — the incumbent
    # hits the tenure/accommodation cells to ~1% and ours was off ±30%.
    from microcosm.build.uk_runtime.lcfs_consumption import rake_energy_to_need

    rng = np.random.default_rng(3)
    n = 400
    household = pd.DataFrame(
        {
            "household_gross_income": rng.uniform(5e3, 2e5, n),
            "electricity_consumption": rng.uniform(200.0, 2000.0, n),
            "gas_consumption": rng.uniform(100.0, 1500.0, n),
        }
    )
    tenure = rng.choice(["OWNED_OUTRIGHT", "RENT_PRIVATELY", "RENT_FROM_COUNCIL"], n)
    accommodation = rng.choice(["HOUSE_DETACHED", "FLAT", "OTHER"], n)
    region = rng.choice(["LONDON", "WALES", "SCOTLAND"], n)
    weights = rng.uniform(0.5, 2.0, n)

    raked = rake_energy_to_need(
        household,
        weights=weights,
        tenure=tenure,
        accommodation=accommodation,
        region=region,
    )

    import json as json_module
    from importlib.resources import files

    need = json_module.loads(
        files("microcosm.build.uk")
        .joinpath("need_energy_targets.json")
        .read_text(encoding="utf-8")
    )
    rates = need["source"]["ofgem_q2_2026"]

    def wmean(values, mask):
        return float((values[mask] * weights[mask]).sum() / weights[mask].sum())

    # Region is the last margin swept, so it fits essentially exactly; the
    # earlier margins settle within a tight band over 50 iterations.
    elec = raked["electricity_consumption"].to_numpy(dtype=float)
    target = (
        need["region"]["electricity_kwh"]["LONDON"] * (rates["electricity_gbp_per_kwh"])
    )
    assert abs(wmean(elec, region == "LONDON") - target) / target < 1e-6
    gas = raked["gas_consumption"].to_numpy(dtype=float)
    tenure_target = need["tenure"]["gas_kwh"]["owner"] * rates["gas_gbp_per_kwh"]
    assert (
        abs(wmean(gas, tenure == "OWNED_OUTRIGHT") - tenure_target) / tenure_target
        < 0.02
    )
    accomm_target = (
        need["accommodation"]["electricity_kwh"]["detached"]
        * rates["electricity_gbp_per_kwh"]
    )
    assert (
        abs(wmean(elec, accommodation == "HOUSE_DETACHED") - accomm_target)
        / accomm_target
        < 0.02
    )
    # Unmapped categories stay outside their margin: SCOTLAND has no NEED
    # region row and OTHER has no accommodation row, but both still move via
    # the other margins — assert they were not pinned to any region target.
    scotland_mean = wmean(elec, region == "SCOTLAND")
    for kwh in need["region"]["electricity_kwh"].values():
        assert abs(scotland_mean - kwh * rates["electricity_gbp_per_kwh"]) > 1.0
