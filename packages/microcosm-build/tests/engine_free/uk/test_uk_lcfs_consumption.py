from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.lcfs_consumption import (
    BUS_FARE_LCFS_CODES,
    LCFS_ACCOMM_MAP,
    LCFS_REGIONS,
    LCFS_TENURE_MAP,
    UK_LCFS_CONSUMPTION_PREDICTORS,
    UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
    UKLCFSConsumptionResult,
    UKLCFSConsumptionStageTransform,
    assign_recipient_has_fuel,
    clean_lcfs_consumption_table,
    derive_energy_from_lcfs,
    impute_lcfs_consumption,
    recipient_predictors,
    support_clip_to_donor,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame

#: About the UK's households, for synthetic frames the DESNZ energy level reads.
UK_HOUSEHOLDS = 28_500_000.0


def _energy_codes(n: int, **values) -> dict[str, np.ndarray]:
    """Every labelled LCFS energy code at zero, with the given ones set."""

    from microcosm.build.uk_runtime.lcfs_consumption import UK_LCFS_ENERGY_CODES

    codes = {code: np.zeros(n) for code in UK_LCFS_ENERGY_CODES if code != "p537"}
    codes.update({code: np.asarray(v, dtype=float) for code, v in values.items()})
    return codes


def _household() -> pd.DataFrame:
    base = {
        "case": [1, 2, 3, 4],
        "g018": [2, 1, 3, 1],
        "g019": [0, 1, 2, 0],
        "gorx": [7, 12, 10, 1],
        "a124": [0, 1, 2, 9],
        "p389p": [100.0, 200.0, 300.0, 400.0],
        "p344p": [150.0, 250.0, 350.0, 450.0],
        "cal_weight_annual": [1.5, 2.0, 2.5, 3.0],
        "a122": [4, 8, 5, 7],
        "a121": [4, 5, 6, 7],
        **_energy_codes(
            4,
            b231=[10.0, 0.0, 0.0, 0.0],
            b226=[4.0, 0.0, 0.0, 0.0],
            b227=[6.0, 0.0, 0.0, 0.0],
            b175=[0.0, 9.0, 0.0, 0.0],
            b178=[0.0, 1.0, 0.0, 0.0],
            b489=[0.0, 8.0, 0.0, 0.0],
            b170=[0.0, 5.0, 0.0, 0.0],
            b490=[0.0, 5.0, 0.0, 0.0],
            b1751=[0.0, 0.0, 7.0, 0.0],
            b017=[0.0, 0.0, 3.0, 0.0],
            b233=[0.0, 0.0, 0.0, 10.0],
        ),
        "p537": [10.0, 13.0, 10.0, 10.0],
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


@pytest.mark.parametrize(
    ("column", "codes", "named"),
    [
        ("gorx", [7, 12, 13, 1], "13"),
        ("a122", [4, 8, 9, 7], "9"),
        ("a121", [4, 5, 6, 0], "0"),
        ("a122", [4, 8, " ", 7], "blank or non-numeric"),
    ],
)
def test_lcfs_donor_cleaning_refuses_unmapped_codes(column, codes, named) -> None:
    # microcosm#1164: a recoded survey year must stop the stage, not lose donors.
    household = _household()
    household[column] = codes

    with pytest.raises(ValueError, match=f"LCFS {column} has code") as raised:
        clean_lcfs_consumption_table(_person(), household)

    assert named in str(raised.value)
    # The refusal names codes, never a count of rows.
    assert "row" not in str(raised.value)


def test_mapped_lcfs_codes_names_every_unknown_code_once() -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import mapped_lcfs_codes

    mapped = mapped_lcfs_codes(pd.Series([1, 12]), LCFS_REGIONS, column="gorx")
    assert mapped.tolist() == ["NORTH_EAST", "NORTHERN_IRELAND"]
    with pytest.raises(ValueError) as raised:
        mapped_lcfs_codes(
            pd.Series([14, 13, 13, None, "x"]), LCFS_REGIONS, column="gorx"
        )
    assert str(raised.value) == (
        "LCFS gorx has code(s) the stage does not map: 13, 14, blank or non-numeric."
    )


def test_stage_declares_the_unmapped_code_refusal() -> None:
    import dataclasses

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        require_refused_unmapped_codes,
    )

    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    require_refused_unmapped_codes(stage)
    operations = tuple(
        dataclasses.replace(
            operation,
            parameters={
                key: value
                for key, value in operation.parameters.items()
                if key != "unmapped_codes"
            },
        )
        if operation.kind == "derive"
        else operation
        for operation in stage.operations
    )
    with pytest.raises(ValueError, match="unmapped_codes"):
        require_refused_unmapped_codes(
            dataclasses.replace(stage, operations=operations)
        )


def test_energy_split_sums_each_fuels_labelled_parts() -> None:
    # microcosm#1164: b226 is the gas component of a combined bill and b227 the
    # electricity one; b489 is the electricity account less its rebate (b175
    # less b178), b490 the gas one (b170 less b173); b1751 the electricity
    # prepayment meter.
    split = derive_energy_from_lcfs(_household())

    electricity = split["electricity_consumption"].to_numpy()
    gas = split["gas_consumption"].to_numpy()
    np.testing.assert_allclose(electricity[:3], [6.0, 8.0, 7.0])
    np.testing.assert_allclose(gas[:3], [4.0, 5.0, 0.0])
    # The unsplit combined key/meter payment takes the dual-fuel mean share.
    mean_share = np.mean([6.0 / 10.0, 8.0 / 13.0])
    np.testing.assert_allclose(electricity[3], 10.0 * mean_share)
    np.testing.assert_allclose(gas[3], 10.0 * (1.0 - mean_share))
    # Oil is neither: it stays in the housing total.
    np.testing.assert_allclose(split["lcfs_other_fuels"], [0.0, 0.0, 3.0, 0.0])


def test_energy_split_shares_an_unsplit_payment_by_the_households_own_split() -> None:
    household = _household()
    # A combined bill only partly split: 2 of electricity and 2 of gas are
    # known, 6 are not, and go 50:50.
    household.loc[0, ["b231", "b227", "b226"]] = [10.0, 2.0, 2.0]

    split = derive_energy_from_lcfs(household)

    assert split.loc[0, "electricity_consumption"] == pytest.approx(5.0)
    assert split.loc[0, "gas_consumption"] == pytest.approx(5.0)


def test_energy_split_gives_a_combined_payment_both_fuels() -> None:
    # Vahid's review of microcosm#1260: a combined key/meter payment means the
    # household pays for both fuels, so a diary top-up coded to electricity
    # alone does not make the whole payment electricity; the unsplit payment
    # takes the dual-fuel mean share, not the household's own share of 1.
    household = _household()
    household.loc[3, "c45114"] = 1.0

    split = derive_energy_from_lcfs(household)

    mean_share = np.mean([6.0 / 10.0, 8.0 / 13.0])
    assert split.loc[3, "electricity_consumption"] == pytest.approx(
        1.0 + 10.0 * mean_share
    )
    assert split.loc[3, "gas_consumption"] == pytest.approx(10.0 * (1.0 - mean_share))


def test_energy_split_nets_the_combined_bill_rebate() -> None:
    # b174 is the combined gas-and-electricity rebate (Volume F Part 2 row 286):
    # an unsplit combined bill of 10 with a rebate of 2 leaves 8 to share.
    household = _household()
    household.loc[0, ["b231", "b226", "b227", "b174"]] = [10.0, 0.0, 0.0, 2.0]

    split = derive_energy_from_lcfs(household)

    total = split.loc[0, "electricity_consumption"] + split.loc[0, "gas_consumption"]
    assert total == pytest.approx(8.0)


def test_energy_split_refuses_an_account_that_is_not_payment_less_rebate() -> None:
    # The donor must show b489 = b175 - b178 for nearly every account payer:
    # a rebate stored with the other sign, or a moved label, breaks it.
    household = _household()
    household.loc[1, "b489"] = 10.0  # b175 + b178, not b175 - b178

    with pytest.raises(ValueError, match="electricity account less rebate"):
        derive_energy_from_lcfs(household)


def test_energy_split_refuses_a_donor_with_no_dual_fuel_household() -> None:
    household = _household()
    household.loc[1, ["b170", "b490"]] = [0.0, 0.0]
    household.loc[0, "b226"] = 0.0

    with pytest.raises(ValueError, match="pays for both electricity and gas"):
        derive_energy_from_lcfs(household)


def test_energy_split_reads_negative_payments_as_zero() -> None:
    household = _household()
    # A refund larger than the account, and a negative meter payment.
    household.loc[1, ["b175", "b178", "b489"]] = [2.0, 5.0, -3.0]
    household.loc[2, "b1751"] = -4.0

    split = derive_energy_from_lcfs(household)

    assert split.loc[1, "electricity_consumption"] == 0.0
    assert split.loc[1, "gas_consumption"] == pytest.approx(5.0)
    assert split.loc[2, "electricity_consumption"] == 0.0
    assert (split[["electricity_consumption", "gas_consumption"]] >= 0).all().all()


def test_energy_split_refuses_a_missing_labelled_code() -> None:
    with pytest.raises(ValueError, match="b2332"):
        derive_energy_from_lcfs(_household().drop(columns="b2332"))


def test_donor_records_its_fuel_light_and_power_split() -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_ENERGY_SPLIT_RECEIPT,
    )

    donor = clean_lcfs_consumption_table(_person(), _household())
    receipt = donor.attrs[UK_LCFS_ENERGY_SPLIT_RECEIPT]
    weights = np.array([1.5, 2.0, 2.5, 3.0]) * 1_000 * (365.25 / 7)
    mean_share = np.mean([6.0 / 10.0, 8.0 / 13.0])
    electricity = np.array([6.0, 8.0, 7.0, 10.0 * mean_share])
    assert receipt["electricity"] == pytest.approx(float(weights @ electricity))
    assert receipt["other_fuels"] == pytest.approx(float(weights[2] * 3.0))
    assert receipt["fuel_light_and_power"] == pytest.approx(
        float(weights @ np.array([10.0, 13.0, 10.0, 10.0]))
    )
    # The labelled parts reach the whole of p537 on this donor.
    assert receipt["parts_over_fuel_light_and_power"] == pytest.approx(1.0)
    assert receipt["clipping_over_fuel_light_and_power"] == pytest.approx(0.0)
    assert receipt["unclipped_over_fuel_light_and_power"] == pytest.approx(1.0)
    assert receipt["share_of_donors_whose_unclipped_parts_differ_from_p537"] == {
        "by_over_a_penny_a_week": 0.0,
        "by_over_50p_a_week": 0.0,
    }
    assert receipt["account_identity_shares"] == {"electricity": 1.0, "gas": 1.0}


def test_split_receipt_separates_refunds_read_as_zero_from_other_gaps() -> None:
    # Vahid's review of microcosm#1260: how much of the parts' excess over p537
    # is refunds read as zero, and how many donors differ from p537 otherwise.
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_ENERGY_SPLIT_RECEIPT,
    )

    household = _household()
    household.loc[2, "b1751"] = -4.0  # a meter refund, read as zero
    household.loc[0, "p537"] = 10.3  # p537 30p a week above the parts

    receipt = clean_lcfs_consumption_table(_person(), household).attrs[
        UK_LCFS_ENERGY_SPLIT_RECEIPT
    ]

    weights = np.array([1.5, 2.0, 2.5, 3.0]) * 1_000 * (365.25 / 7)
    p537 = np.array([10.3, 13.0, 10.0, 10.0])
    unclipped = np.array([10.0, 13.0, -4.0 + 3.0, 10.0])
    assert receipt["clipping_over_fuel_light_and_power"] == pytest.approx(
        float(weights[2] * 4.0 / (weights @ p537))
    )
    assert receipt["unclipped_over_fuel_light_and_power"] == pytest.approx(
        float(weights @ unclipped / (weights @ p537))
    )
    # Two donors differ: a small cell, so no share.
    assert receipt["share_of_donors_whose_unclipped_parts_differ_from_p537"] == {
        "by_over_a_penny_a_week": "<10",
        "by_over_50p_a_week": "<10",
    }


def test_split_receipt_shares_differing_donors_by_threshold() -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import (
        donor_energy_split_receipt,
    )

    household = pd.concat([_household()] * 5, ignore_index=True)
    household.loc[:11, "p537"] += 1.0  # twelve donors a pound a week off
    household.loc[12:13, "p537"] += 0.2  # two only 20p off
    split = derive_energy_from_lcfs(household)
    split["household_weight"] = split["cal_weight_annual"]

    receipt = donor_energy_split_receipt(split)

    weight = household["cal_weight_annual"].to_numpy()
    assert receipt["share_of_donors_whose_unclipped_parts_differ_from_p537"] == {
        "by_over_a_penny_a_week": pytest.approx(weight[:14].sum() / weight.sum()),
        "by_over_50p_a_week": pytest.approx(weight[:12].sum() / weight.sum()),
    }


def test_split_receipt_floors_the_account_identity_shares() -> None:
    # An exact unweighted share gives back its counts: the receipt floors it.
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_ACCOUNT_IDENTITY_RECEIPT,
    )

    household = pd.concat([_household()] * 40, ignore_index=True)
    # One electricity account payer in forty whose b489 is not b175 - b178.
    household.loc[1, "b489"] = 9.0

    split = derive_energy_from_lcfs(household)

    assert split.attrs[UK_LCFS_ACCOUNT_IDENTITY_RECEIPT] == {
        "electricity": 0.97,
        "gas": 1.0,
    }


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


def test_fuel_flag_draws_each_vehicle_at_the_zero_emission_share() -> None:
    """microcosm#1278: a household with n vehicles buys fuel unless all n are
    zero-emission, so it is flagged at 1 - (1 - rate) ** n against its keyed
    uniform; one-vehicle households keep the draw of the per-household rule."""

    from microcosm.build.stochastic_assignment import stable_identity_uniforms

    ids = np.arange(1, 4_001)
    vehicles = np.tile([0, 1, 2, 3], 1_000)
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": np.ones(len(ids)),
                "num_vehicles": vehicles,
            }
        ),
        time_period="2023",
    )
    rate = 0.6
    flagged = assign_recipient_has_fuel(frame, rate=rate, seed=0)
    draws = stable_identity_uniforms(ids, seed=0, salt="lcfs_has_fuel_consumption")
    expected = (vehicles > 0) & (draws < 1.0 - (1.0 - rate) ** vehicles)
    assert flagged.tolist() == expected.tolist()
    assert not flagged[vehicles == 0].any()
    # One vehicle: exactly the old per-household draw at the rate.
    one = vehicles == 1
    assert flagged[one].tolist() == (draws[one] < rate).tolist()
    # More vehicles, more fuel buyers: about 1 - 0.4 ** n of each group.
    for n in (1, 2, 3):
        share = flagged[vehicles == n].mean()
        assert share == pytest.approx(1.0 - (1.0 - rate) ** n, abs=0.05)
    with pytest.raises(ValueError, match="fuel-buyer share"):
        assign_recipient_has_fuel(frame, rate=0.0, seed=0)


def test_negative_donor_consumption_is_floored_with_a_receipt() -> None:
    """microcosm#1063 c9: a diary spend netting refunds below zero is raised to
    the declared floor before the clip ranges and the imputation see it."""
    from microcosm.build.uk_runtime.lcfs_consumption import (
        CONSUMPTION_VARIABLE_RENAMES,
        UK_LCFS_DONOR_FLOORED_COLUMNS,
        floor_negative_donor_consumption,
    )

    # The bus fares, summed from three diary lines, are floored too.
    columns = list(UK_LCFS_DONOR_FLOORED_COLUMNS)
    assert columns == [*CONSUMPTION_VARIABLE_RENAMES.values(), "bus_fare_spending"]
    donor = pd.DataFrame({column: [10.0, 20.0, 30.0] for column in columns})
    donor.loc[1, "housing_water_and_electricity_consumption"] = -50.0
    donor.loc[2, "housing_water_and_electricity_consumption"] = -1.0
    donor.loc[0, "bus_fare_spending"] = -2.0
    raised = {"housing_water_and_electricity_consumption", "bus_fare_spending"}

    floored, receipt = floor_negative_donor_consumption(donor)

    assert floored["housing_water_and_electricity_consumption"].tolist() == [
        10.0,
        0.0,
        0.0,
    ]
    assert floored["bus_fare_spending"].tolist() == [0.0, 20.0, 30.0]
    for column in columns:
        if column not in raised:
            assert floored[column].tolist() == [10.0, 20.0, 30.0]
    assert receipt["floor"] == 0.0
    # Counts of rows picked out by a negative spend are banded (microcosm#1164).
    assert receipt["rows_raised"] == "<10"
    assert receipt["remaining_negative_rows"] == 0
    housing = receipt["columns"]["housing_water_and_electricity_consumption"]
    # Counts only: no donor value or statistic of the raised rows.
    assert housing == {"rows_raised": "<10"}
    assert receipt["columns"]["bus_fare_spending"] == {"rows_raised": "<10"}
    assert receipt["columns"]["food_and_non_alcoholic_beverages_consumption"] == {
        "rows_raised": 0
    }
    with pytest.raises(ValueError, match="missing consumption column"):
        floor_negative_donor_consumption(donor.drop(columns=[columns[0]]))
    with pytest.raises(ValueError, match="bus_fare_spending"):
        floor_negative_donor_consumption(donor.drop(columns=["bus_fare_spending"]))


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
    assert receipt["rule"] == "per_car_zero_emission_share"
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


def test_recipient_need_band_income_is_hbai_gross_income() -> None:
    # microcosm#1164: the donor bands on gross p344p, so the recipient bands on
    # its HBAI net income with the engine's HBAI deductions added back, neither
    # the (lower) net income nor the engine's household_gross_income, which
    # counts capital gains and in-kind NHS and education spending.
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.household_income import (
        UK_HBAI_GROSS_INCOME_DEDUCTIONS,
    )
    from microcosm.build.uk_runtime.lcfs_consumption import recipient_predictors

    class _Engine:
        def materialize(self, frame, variables, period):
            assert "household_gross_income" not in variables
            assert set(UK_HBAI_GROSS_INCOME_DEDUCTIONS) <= set(variables)
            values = {
                "hbai_household_net_income_before_reset_to_zero": np.array(
                    [20_000.0, 50_000.0]
                ),
                "income_tax": np.array([3_000.0, 20_000.0]),
                "council_tax": np.array([1_500.0, 2_500.0]),
                "national_insurance": np.array([500.0, 7_500.0]),
            }
            return {name: values.get(name, np.array([0.0, 0.0])) for name in variables}

        def variable_metadata(self, name):
            return SimpleNamespace(entity="household")

    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_household_id": [10, 20],
                "age": [40.0, 70.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame(
            {
                "household_id": [10, 20],
                "household_weight": [1.0, 1.0],
                "num_vehicles": [1, 0],
            }
        ),
        time_period="2024",
    )

    predictors = recipient_predictors(frame, _Engine())

    assert predictors["household_gross_income"].tolist() == [25_000.0, 80_000.0]
    assert predictors["hbai_household_net_income"].tolist() == [20_000.0, 50_000.0]
    assert not set(UK_HBAI_GROSS_INCOME_DEDUCTIONS) & set(predictors.columns)


def test_recipient_predictors_read_the_vehicle_count_numerically() -> None:
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.household_income import (
        UK_HBAI_GROSS_INCOME_DEDUCTIONS,
        UK_HBAI_NET_INCOME_PREDICTOR,
        UK_HBAI_NET_INCOME_VARIABLE,
    )
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
    # The HBAI deductions only build the NEED band's gross income, and the HBAI
    # net income, read before the reset to zero, keeps its predictor name.
    engine_columns = {
        UK_HBAI_NET_INCOME_PREDICTOR if name == UK_HBAI_NET_INCOME_VARIABLE else name
        for name in UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS
        if name not in UK_HBAI_GROSS_INCOME_DEDUCTIONS
    }
    assert engine_columns | {"household_gross_income"} <= set(predictors.columns)
    assert UK_HBAI_NET_INCOME_VARIABLE not in predictors.columns
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

    # Ten households or more in every group on both sides.
    donor = pd.DataFrame(
        {
            "household_weight": [1.0] * 20 + [2.0] * 20,
            "num_vehicles": [0] * 10 + [1] * 30,
            "petrol_spending": [0.0] * 20 + [10.0] * 20,
            "diesel_spending": [0.0] * 40,
        }
    )
    recipient = pd.DataFrame(
        {
            "num_vehicles": [0] * 10 + [1] * 20,
            "has_fuel_consumption": [False] * 10 + [True] * 10 + [False] * 10,
        }
    )
    evidence = fuel_flag_evidence(
        donor,
        recipient,
        recipient_weights=[1.0] * 20 + [2.0] * 10,
        ice_share_receipt={"rate": 0.9},
    )
    assert evidence["ice_share"] == {"rate": 0.9}
    assert evidence["donor"] == {
        "households": 40,
        "with_vehicles_share": pytest.approx(50 / 60),
        "positive_fuel_share": pytest.approx(40 / 60),
        "positive_fuel_share_among_vehicle_households": pytest.approx(40 / 50),
    }
    assert evidence["recipient"] == {
        "households": 30,
        "with_vehicles_share": pytest.approx(0.75),
        "flagged_share": pytest.approx(0.25),
        "flagged_share_among_vehicle_households": pytest.approx(1 / 3),
        "fuel_free_share_of_vehicles": pytest.approx(2 / 3),
    }


def test_fuel_flag_evidence_withholds_every_share_beside_a_thin_group() -> None:
    """The shares on one side difference into the households with and
    without a vehicle, crossed with fuel or none, so one group on one to
    nine sources withholds every share on that side (microcosm#1306 review)."""

    from microcosm.build.uk_runtime.lcfs_consumption import fuel_flag_evidence

    donor = pd.DataFrame(
        {
            "household_weight": [1.0, 1.0, 2.0],
            "num_vehicles": [0, 1, 2],
            "petrol_spending": [0.0, 0.0, 10.0],
            "diesel_spending": [0.0, 0.0, 0.0],
        }
    )
    # Twenty buyers and ten households without a vehicle, but three
    # fuel-free vehicle households.
    recipient = pd.DataFrame(
        {
            "num_vehicles": [0] * 10 + [1] * 23,
            "has_fuel_consumption": [False] * 10 + [True] * 20 + [False] * 3,
        }
    )
    evidence = fuel_flag_evidence(
        donor,
        recipient,
        recipient_weights=np.ones(33),
        ice_share_receipt={"rate": 0.9},
    )
    assert evidence["donor"] == {
        "households": "<10",
        "with_vehicles_share": "<10",
        "positive_fuel_share": "<10",
        "positive_fuel_share_among_vehicle_households": "<10",
    }
    assert evidence["recipient"] == {
        "households": 33,
        "with_vehicles_share": "<10",
        "flagged_share": "<10",
        "flagged_share_among_vehicle_households": "<10",
        "fuel_free_share_of_vehicles": "<10",
    }


def test_fuel_flag_evidence_counts_the_cars_in_fuel_free_households() -> None:
    """The flag is per household, so a household with a petrol car beside a
    zero-emission one buys fuel: the receipt sets the fuel-free share of
    vehicles beside the per-car share (microcosm#1306 review)."""

    from microcosm.build.uk_runtime.lcfs_consumption import fuel_flag_evidence

    donor = pd.DataFrame(
        {
            "household_weight": [1.0],
            "num_vehicles": [1],
            "petrol_spending": [10.0],
            "diesel_spending": [0.0],
        }
    )
    recipient = pd.DataFrame(
        {
            "num_vehicles": [1] * 12 + [2] * 10 + [1] * 10 + [0] * 12,
            "has_fuel_consumption": [True] * 22 + [False] * 22,
        }
    )
    evidence = fuel_flag_evidence(
        donor,
        recipient,
        recipient_weights=np.ones(44),
        ice_share_receipt={"rate": 0.96},
    )["recipient"]
    assert evidence["flagged_share_among_vehicle_households"] == pytest.approx(22 / 32)
    # Forty-two vehicles, ten of them in the ten fuel-free households.
    assert evidence["fuel_free_share_of_vehicles"] == pytest.approx(10 / 42)
    # Copies of one source household count once: the ten fuel-free rows are
    # copies of three sources, so every share is withheld.
    sources = list(range(22)) + [100, 100, 100, 100, 101, 101, 101, 102, 102, 102]
    sources += list(range(200, 212))
    withheld = fuel_flag_evidence(
        donor,
        recipient,
        recipient_weights=np.ones(44),
        ice_share_receipt={"rate": 0.96},
        sources=sources,
    )["recipient"]
    assert withheld["fuel_free_share_of_vehicles"] == "<10"
    assert withheld["flagged_share"] == "<10"


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
        GasPayerDraws,
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
    # Weights summing to the UK's households, so the DESNZ level gives each
    # household a plausible kWh and the domestic ceilings bind only on outliers.
    weights = rng.uniform(0.5, 2.0, n)
    weights *= UK_HOUSEHOLDS / weights.sum()

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
        gas_payers=GasPayerDraws(
            probability=np.full(n, 0.5),
            positive_spend=np.full(n, 600.0),
            receipt={"rule": "gas_payer_probability_order"},
            fit_weight_records=(),
        ),
    )
    assert receipt["gas_connection"]["disconnect_rule"] == "identity_uniform_order"
    assert receipt["gas_connection"]["seed"] == 0
    assert receipt["gas_connection"]["connect_rule"] == "gas_payer_probability_order"
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
        "cal_weight_annual": rng.uniform(0.5, 2.0, n),
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
    # The labelled energy parts (microcosm#1164), deterministic so the draws
    # below keep their stream: an electricity account everywhere, a gas account
    # on five households in six, oil on a few.
    household.update(
        _energy_codes(
            n,
            b175=20.0 + rows % 15,
            b489=20.0 + rows % 15,
            b170=np.where(rows % 6 == 0, 0.0, 10.0 + rows % 9),
            b490=np.where(rows % 6 == 0, 0.0, 10.0 + rows % 9),
            b226=household["b226"],
            b017=np.where(rows % 20 == 3, 5.0, 0.0),
        )
    )
    person = pd.DataFrame(
        {
            "case": rows + 1,
            "b303p": rng.uniform(0.0, 900.0, n),
            "b3262p": rng.uniform(0.0, 100.0, n),
            "p049p": rng.uniform(0.0, 200.0, n),
        }
    )
    return person, pd.DataFrame(household)


def _synthetic_lcfs_stage(household_columns=None):
    """The committed lcfs_consumption stage on synthetic inputs, minus the engine.

    The uprating step needs the installed engine and is dropped; the QRF is
    shrunk to four trees. ``household_columns(n)`` adds frame columns.
    """

    import dataclasses
    from types import SimpleNamespace

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.source_manifest import SourceOperationSpec

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
            # Summing to about the UK's households, so the DESNZ energy level
            # gives each household a plausible kWh under the domestic ceilings.
            "household_weight": rng.uniform(0.5, 3.0, n) * (UK_HOUSEHOLDS / (1.75 * n)),
            "region": regions,
            "tenure_type": np.array(
                ["OWNED_OUTRIGHT", "RENT_PRIVATELY", "RENT_FROM_COUNCIL"]
            )[np.arange(n) % 3],
            "accommodation_type": np.array(
                ["HOUSE_DETACHED", "FLAT", "HOUSE_TERRACED"]
            )[np.arange(n) % 3],
            "num_vehicles": np.arange(n) % 3,
            "household_gross_income": rng.uniform(5e3, 9e4, n),
            **(household_columns(n) if household_columns else {}),
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
                "hbai_household_net_income_before_reset_to_zero": frame.table(
                    "household"
                )["household_gross_income"].to_numpy()
                * 0.8,
                # The NEED band adds the HBAI deductions back (microcosm#1164).
                "income_tax": frame.table("household")[
                    "household_gross_income"
                ].to_numpy()
                * 0.2,
            }
            return {name: values.get(name, np.zeros(rows)) for name in variables}

    donor_person, donor_household = _synthetic_lcfs_donor()
    transform = UKLCFSConsumptionStageTransform(
        stage=stage,
        engine=_Engine(),
        lcfs_household=donor_household,
        lcfs_person=donor_person,
    )
    return SimpleNamespace(
        transform=transform,
        frame=frame,
        household=household,
        person=person,
        regions=regions,
        n=n,
        donor_household=donor_household,
    )


def test_stage_transform_prices_bus_fares_from_journeys() -> None:
    """End to end on synthetic inputs: the committed declaration minus the engine.

    The uprating step needs the installed engine and is dropped; the QRF is
    shrunk to four trees. Everything else is the committed lcfs_consumption
    stage (#890 A, E2; #930 pricing).
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS,
        UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS,
    )

    synthetic = _synthetic_lcfs_stage()
    transform, frame, household = (
        synthetic.transform,
        synthetic.frame,
        synthetic.household,
    )
    person, regions, n = synthetic.person, synthetic.regions, synthetic.n
    donor_household = synthetic.donor_household
    result = transform(frame)

    out = result.table("household")
    assert set(UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS) <= set(out.columns)
    # The 20 chain targets (the heating fuels last, pe-uk#2572), the road-fuel
    # redraw's total and petrol share, the prepayment-meter flag
    # (microcosm#1110), then the gas-payer classifier and amount model the
    # connection walk reads (microcosm#1164).
    assert len(transform.fit_weight_records) == 25
    assert transform.fit_weight_records[-4].fit_name.endswith(
        ":petrol_share_of_road_fuel"
    )
    assert transform.fit_weight_records[-3].fit_name.endswith(
        ":uses_energy_prepayment_meter"
    )
    assert transform.fit_weight_records[-2].fit_name.endswith(":gas_payer")
    assert transform.fit_weight_records[-1].fit_name.endswith(":gas_payer_amount")
    assert out["uses_energy_prepayment_meter"].dtype == bool
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_CONSUMPTION_MASS_CONSERVATION_REASON,
    )

    receipt = result.mass_log[-1]
    assert receipt.reason == UK_LCFS_CONSUMPTION_MASS_CONSERVATION_REASON
    assert receipt.old_total == receipt.new_total > 0
    assert receipt.declared_factor == 1.0
    materialize = next(
        operation
        for operation in transform.stage.operations
        if operation.kind == "materialize_rules_engine_predictors"
    )
    assert (
        tuple(materialize.parameters["predictors"])
        == UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS
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
    # The uprating step is dropped in this engine-free run, so no litres audit
    # and no division capture (both read the uprating's target year).
    assert "fuel_litres_audit" not in evidence
    assert "division_capture" not in evidence
    assert "donor_uprating" not in evidence
    # Fuel: no vehicles means no fuel; the fuel-buyer share came from VEH1103.
    no_vehicle = household["num_vehicles"].to_numpy() == 0
    assert (out.loc[no_vehicle, "petrol_spending"] == 0.0).all()
    assert 0.95 < evidence["has_fuel_consumption"]["ice_share"]["rate"] < 0.97
    # Road fuel: petrol plus diesel at prior weights is ONS 07.2.2 for 2024
    # less the donor's own other-fuels share (#1113).
    level = evidence["road_fuel_level"]
    assert level["coicop"] == "07.2.2" and level["period_value"] == 2024
    weight = donor_household["cal_weight_annual"].to_numpy()
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
    # Transport holds the levelled road fuel and the priced bus fares together.
    bus = out["bus_fare_spending"].to_numpy(dtype=float)
    assert (out["transport_consumption"] >= road_fuel + bus - 1e-9).all()
    recomposed = evidence["recomposed_totals"]["parents"]
    household_weights = result.weights_for("household").values
    assert recomposed["transport_consumption"]["weighted_components"] == (
        pytest.approx(level["level"] + household_weights @ bus)
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
    # Every spend level is on calendar 2024; this run drops the uprating step,
    # so the basis gate fails on exactly that receipt.
    basis = uk_stage_health_gate(
        evidence=evidence,
        stage="lcfs_consumption",
        check="consumption_basis",
        parameters={
            "stage": "lcfs_consumption",
            "check": "consumption_basis",
            "period_type": "calendar_year",
            "period_value": 2024,
        },
    )
    assert not basis.passed
    assert list(basis.failures) == [
        "lcfs_consumption: the stage recorded no donor_uprating receipt."
    ], basis.failures
    assert evidence["energy_pricing"]["level"]["period_type"] == "calendar_year"
    assert evidence["bus_pricing"]["price_year"] == 2024
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
    # Each fuel's benchmark is its HMRC litres times the cars share of that
    # fuel's road use (DESNZ, by vehicle and fuel; PolicyEngine/chronicle#322).
    desnz = vendored_rows(
        "desnz_road_fuel_by_vehicle.json",
        concept="desnz.road_transport.fuel_consumption",
        period_type="calendar_year",
        period_value=2024,
        geography_id="K02000001",
    )
    petrol_rows = {
        r["dimensions"]["vehicle_type"]: float(r["value"])
        for r in desnz
        if r["dimensions"]["fuel"] == "petrol"
    }
    share = petrol_rows["cars"] / sum(petrol_rows.values())
    assert petrol["desnz_cars_share"] == pytest.approx(share)
    assert petrol["cars_litres_benchmark"] == pytest.approx(hmrc * share)
    assert petrol["frame_over_cars_benchmark"] == pytest.approx(
        petrol["frame_litres"] / petrol["cars_litres_benchmark"]
    )
    diesel = audit["fuels"]["diesel_spending"]
    # Diesel is mostly vans and lorries, so its cars share is far below petrol's.
    assert diesel["desnz_cars_share"] < petrol["desnz_cars_share"]
    assert set(audit["fuels"]) == {"petrol_spending", "diesel_spending"}
    assert audit["gated"] is False
    assert audit["per_fuel_ratio_basis"] == "desnz cars share of each fuel's road use"
    assert audit["frame_over_obr_uniform_cars_benchmark"] == pytest.approx(
        audit["frame_litres_total"]
        / (
            audit["obr_fuel_duty_receipts"]["cars_share"]
            * (hmrc + diesel["hmrc_litres_all_road_users"])
        )
    )

    class Stage:
        operations = ()

    assert fuel_litres_audit(draws, weights=weights, stage=Stage()) is None


def test_division_capture_compares_each_division_with_ons() -> None:
    """Frame divisions against Consumer Trends less what diaries miss (#1113)."""

    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_COICOP_DIVISION_COLUMNS,
        division_capture,
    )
    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    def ons(coicop: str, sheet: str, concept: str = "domestic") -> float:
        (row,) = vendored_rows(
            "ons_household_expenditure_facts.json",
            concept="ons.household_final_consumption_expenditure",
            period_type="calendar_year",
            period_value=2024,
            dimensions={
                "coicop": coicop,
                "source_sheet": sheet,
                "consumption_concept": concept,
                "price_basis": "current_prices",
                "seasonal_adjustment": "not_seasonally_adjusted",
            },
        )
        return float(row["value"])

    draws = pd.DataFrame(
        {column: [1.0e3, 2.0e3] for column in UK_LCFS_COICOP_DIVISION_COLUMNS}
    )
    weights = np.array([1.0e7, 5.0e6])

    capture = division_capture(draws, weights=weights, period_value=2024)

    frame = 1.0e3 * 1.0e7 + 2.0e3 * 5.0e6
    food = capture["divisions"]["food_and_non_alcoholic_beverages_consumption"]
    assert food["coicop"] == "01" and food["out_of_scope"] == {}
    assert food["published_domestic"] == ons("01", "0CN")
    assert food["capture"] == pytest.approx(frame / ons("01", "0CN"))
    housing = capture["divisions"]["housing_water_and_electricity_consumption"]
    assert housing["out_of_scope"] == {"04.2": ons("04.2", "04CN")}
    assert housing["survey_scope"] == pytest.approx(
        ons("04", "0CN") - ons("04.2", "04CN")
    )
    assert set(
        capture["divisions"]["alcohol_and_tobacco_consumption"]["out_of_scope"]
    ) == {"02.3"}
    assert set(capture["divisions"]["miscellaneous_consumption"]["out_of_scope"]) == {
        "12.6.1"
    }
    total = capture["total"]
    assert total["frame"] == pytest.approx(12 * frame)
    assert total["published_national"] == ons("NAT0", "0CN", "national")
    assert total["capture_national"] == pytest.approx(
        12 * frame / (total["published_national"] - total["out_of_scope"])
    )
    assert capture["gated"] is False
    assert "donor" not in food and "donor" not in total


def test_division_capture_reads_each_division_against_its_donor() -> None:
    """The frame over the donor it drew from, in all and per household (#1207)."""

    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_COICOP_DIVISION_COLUMNS,
        division_capture,
    )

    draws = pd.DataFrame(
        {column: [1.0e3, 2.0e3] for column in UK_LCFS_COICOP_DIVISION_COLUMNS}
    )
    weights = np.array([1.0e7, 5.0e6])
    donor = pd.DataFrame(
        {column: [500.0, 1.5e3, 1.0e3] for column in UK_LCFS_COICOP_DIVISION_COLUMNS}
        | {"household_weight": [4.0e6, 6.0e6, 2.0e6]}
    )

    capture = division_capture(draws, weights=weights, period_value=2024, donor=donor)

    frame = 1.0e3 * 1.0e7 + 2.0e3 * 5.0e6
    donor_total = 500.0 * 4.0e6 + 1.5e3 * 6.0e6 + 1.0e3 * 2.0e6
    food = capture["divisions"]["food_and_non_alcoholic_beverages_consumption"]
    assert food["donor"] == pytest.approx(donor_total)
    assert food["frame_over_donor"] == pytest.approx(frame / donor_total)
    assert food["frame_over_donor_per_household"] == pytest.approx(
        (frame / 1.5e7) / (donor_total / 1.2e7)
    )
    total = capture["total"]
    assert total["donor"] == pytest.approx(12 * donor_total)
    assert total["frame_over_donor_per_household"] == pytest.approx(
        food["frame_over_donor_per_household"]
    )


def test_weighted_top_share_withholds_fewer_than_ten_households() -> None:
    """The top spenders' share of a weighted total (microcosm#1278); a share
    resting on one to nine households is "<10", as every receipt count."""

    from microcosm.build.uk_runtime.lcfs_consumption import weighted_top_share

    values = np.array([100.0] * 20 + [1.0] * 180)
    weights = np.ones(200)
    assert weighted_top_share(values, weights, fraction=0.10) == pytest.approx(
        2_000.0 / 2_180.0
    )
    assert weighted_top_share(values, weights, fraction=0.01) == "<10"
    # Weight, not row count, sets the group: at double weight, eleven of the
    # top twenty households carry exactly a tenth of the 220 weight.
    heavy = np.where(values > 1.0, 2.0, 1.0)
    assert weighted_top_share(values, heavy, fraction=0.10) == pytest.approx(
        (11 * 2 * 100.0) / (20 * 2 * 100.0 + 180.0)
    )
    assert weighted_top_share(np.zeros(20), np.ones(20), fraction=0.5) is None
    with pytest.raises(ValueError, match="top fraction"):
        weighted_top_share(values, weights, fraction=0.0)


def test_weighted_top_shares_count_distinct_source_households() -> None:
    """A frame stacks copies of one source household, so the floor counts
    sources, not rows (microcosm#1306 review)."""

    from microcosm.build.uk_runtime.lcfs_consumption import (
        division_tail,
        weighted_top_shares,
    )

    values = np.array([100.0] * 12 + [1.0] * 188)
    weights = np.ones(200)
    # Two source households split into six copies each lead the frame.
    sources = np.array([1] * 6 + [2] * 6 + list(range(3, 191)))
    as_rows = weighted_top_shares(values, weights, fractions=(0.06,))
    assert as_rows[0.06] == pytest.approx(1_200.0 / 1_388.0)
    as_sources = weighted_top_shares(
        values, weights, fractions=(0.06,), sources=sources
    )
    assert as_sources[0.06] == "<10"
    tail = division_tail(values, weights, values, weights, frame_sources=sources)
    # Each top 10% is twenty rows, the twelve leaders and eight more: ten
    # sources on the frame and twenty on the donor (one row per household),
    # so both ship. The frame's top 1% is two copies of one source.
    assert tail["top_10_percent"]["donor"] == pytest.approx(1_208.0 / 1_388.0)
    assert tail["top_10_percent"]["frame"] == pytest.approx(1_208.0 / 1_388.0)
    assert tail["top_1_percent"]["frame"] == "<10"
    with pytest.raises(ValueError, match="align"):
        weighted_top_shares(values, weights, fractions=(0.1,), sources=sources[:5])


def test_weighted_top_shares_withhold_a_difference_of_one_to_nine() -> None:
    """The top 1% leads the top 10%, so the two shares' difference is the
    spend of the households between them: a larger fraction adding one to
    nine sources to a shipped smaller one is withheld (microcosm#1306 review)."""

    from microcosm.build.uk_runtime.lcfs_consumption import weighted_top_shares

    def shares(next_rows: int, next_weight: float) -> dict:
        values = np.array([100.0] * 10 + [50.0] * next_rows + [1.0] * 100)
        weights = np.array([10.0] * 10 + [next_weight] * next_rows + [90.0] * 100)
        return weighted_top_shares(values, weights, fractions=(0.01, 0.10))

    # Ten households carry the top 1% and five more the rest of the top 10%.
    five_more = shares(5, 180.0)
    assert five_more[0.01] == pytest.approx(10_000.0 / 64_000.0)
    assert five_more[0.10] == "<10"
    # Ten more is enough.
    ten_more = shares(10, 90.0)
    assert ten_more[0.01] == pytest.approx(10_000.0 / 64_000.0)
    assert ten_more[0.10] == pytest.approx(55_000.0 / 64_000.0)


def test_weighted_top_shares_withhold_a_thin_rest_and_count_spenders() -> None:
    """The receipt's total ships beside the shares, so total x (1 - share) is
    the spend outside the top group: a share is withheld when one to nine
    sources spend anything outside it. The floors count spenders, so zero
    rows in a group lend it nothing (microcosm#1306 review)."""

    from microcosm.build.uk_runtime.lcfs_consumption import weighted_top_shares

    # Twenty leaders carry the top 10%; five others spend; 175 spend nothing.
    values = np.array([100.0] * 20 + [7.0] * 5 + [0.0] * 175)
    shares = weighted_top_shares(values, np.ones(200), fractions=(0.10,))
    assert shares[0.10] == "<10"
    # With ten spenders outside it, the share ships.
    values = np.array([100.0] * 20 + [7.0] * 10 + [0.0] * 170)
    shares = weighted_top_shares(values, np.ones(200), fractions=(0.10,))
    assert shares[0.10] == pytest.approx(2_000.0 / 2_070.0)
    # A top group of twenty rows of which four spend rests on four spenders.
    values = np.array([100.0] * 4 + [0.0] * 196)
    shares = weighted_top_shares(values, np.ones(200), fractions=(0.10,))
    assert shares[0.10] == "<10"


def test_division_capture_counts_source_households_not_copies() -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_COICOP_DIVISION_COLUMNS,
        division_capture,
    )

    # The twenty leaders are copies of four source households.
    frame_values = [100.0] * 20 + [1.0] * 180
    sources = [index // 5 for index in range(20)] + list(range(100, 280))
    draws = pd.DataFrame(
        {column: frame_values for column in UK_LCFS_COICOP_DIVISION_COLUMNS}
    )
    donor = pd.DataFrame(
        {column: [10.9] * 200 for column in UK_LCFS_COICOP_DIVISION_COLUMNS}
        | {"household_weight": [1.0] * 200}
    )
    column = "food_and_non_alcoholic_beverages_consumption"
    as_rows = division_capture(
        draws, weights=np.ones(200), period_value=2024, donor=donor
    )["divisions"][column]["tail"]["top_10_percent"]
    assert as_rows["frame"] == pytest.approx(2_000.0 / 2_180.0)
    as_sources = division_capture(
        draws, weights=np.ones(200), period_value=2024, donor=donor, sources=sources
    )["divisions"][column]["tail"]["top_10_percent"]
    assert as_sources["frame"] == "<10"
    assert as_sources["donor"] == pytest.approx(0.10)


def test_source_households_refuse_copies_without_their_sources() -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import _source_households

    plain = pd.DataFrame({"household_id": [1, 2]})
    assert _source_households(plain) is None
    named = plain.assign(source_household_id=[7, 7])
    assert list(_source_households(named)) == [7, 7]
    for marker in (
        "household_support_clone_index",
        "household_is_high_value_support_copy",
        "high_value_support_copy_index",
    ):
        with pytest.raises(ValueError, match="source_household_id"):
            _source_households(plain.assign(**{marker: [0, 1]}))


def test_stage_transform_counts_source_households_in_its_receipts(
    monkeypatch,
) -> None:
    """The stage hands each frame row's source household to both receipts
    that floor counts (microcosm#1306 review)."""

    from microcosm.build.uk_runtime import lcfs_consumption as module

    synthetic = _synthetic_lcfs_stage(
        lambda n: {"source_household_id": np.arange(n) // 2}
    )
    seen = {}
    real_flag = module.fuel_flag_evidence

    def flag(*args, **kwargs):
        seen["fuel_flag"] = kwargs["sources"]
        return real_flag(*args, **kwargs)

    def capture(*args, **kwargs):
        seen["division_capture"] = kwargs["sources"]
        return {"available": False, "gated": False}

    # The capture runs only with the uprating step, which needs the engine:
    # skip the donor uprating and hand the capture its year directly.
    monkeypatch.setattr(module, "fuel_flag_evidence", flag)
    monkeypatch.setattr(module, "lcfs_donor_uprating", lambda stage: ({}, None))
    monkeypatch.setattr(module, "uprating_operation", lambda stage: {"to_period": 2024})
    monkeypatch.setattr(module, "fuel_litres_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "division_capture_or_reason", capture)
    synthetic.transform(synthetic.frame)
    expected = synthetic.frame.table("household")["source_household_id"].to_numpy()
    assert np.array_equal(seen["fuel_flag"], expected)
    assert np.array_equal(seen["division_capture"], expected)


def test_division_capture_places_an_over_draw_in_the_tail() -> None:
    """The frame's top 10% hold far more of a division than the donor's do."""

    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_COICOP_DIVISION_COLUMNS,
        division_capture,
    )

    frame_values = [100.0] * 20 + [1.0] * 180
    donor_values = [2_180.0 / 200.0] * 200
    draws = pd.DataFrame(
        {column: frame_values for column in UK_LCFS_COICOP_DIVISION_COLUMNS}
    )
    donor = pd.DataFrame(
        {column: donor_values for column in UK_LCFS_COICOP_DIVISION_COLUMNS}
        | {"household_weight": [1.0] * 200}
    )

    capture = division_capture(
        draws, weights=np.ones(200), period_value=2024, donor=donor
    )

    tail = capture["divisions"]["food_and_non_alcoholic_beverages_consumption"]["tail"]
    top_ten = tail["top_10_percent"]
    assert top_ten["frame"] == pytest.approx(2_000.0 / 2_180.0)
    assert top_ten["donor"] == pytest.approx(0.10)
    assert top_ten["frame_over_donor"] == pytest.approx(2_000.0 / 2_180.0 / 0.10)
    # Two households are the top 1% of two hundred: withheld on both sides.
    assert tail["top_1_percent"] == {
        "frame": "<10",
        "donor": "<10",
        "frame_over_donor": None,
    }
    assert "tail" not in capture["total"]


def test_published_fuels_sit_beside_the_donors_and_the_frames_draws() -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import (
        published_fuels_beside_draws,
    )

    published = {
        "resource": "r",
        "classes": [
            {"coicop": "04.5.3", "column": "liquid_fuel_consumption", "value": 2.0e9},
            {"coicop": "04.5.4", "column": "solid_fuel_consumption", "value": 0.0},
        ],
        "total": 2.0e9,
    }
    beside = published_fuels_beside_draws(
        published,
        {"liquid_fuel_consumption": 1.5e9, "solid_fuel_consumption": 1.0e8},
        {"liquid_fuel_consumption": 1.8e9, "solid_fuel_consumption": 2.0e8},
    )
    liquid, solid = beside["classes"]
    assert liquid["donor_at_diary_prices"] == 1.5e9
    assert liquid["donor_over_published"] == pytest.approx(0.75)
    assert liquid["frame"] == 1.8e9
    assert liquid["frame_over_published"] == pytest.approx(0.9)
    # A published zero gives no ratio.
    assert solid["donor_over_published"] is None
    assert solid["frame_over_published"] is None
    assert beside["total"] == published["total"]


def test_report_only_diagnostics_fail_closed_on_ambiguous_rows(monkeypatch) -> None:
    """A duplicate or ambiguous vendored row is a data fault, not a missing year
    (the #1167 re-review): it raises a plain ValueError that nothing records."""

    from microcosm.build.uk_runtime import lcfs_consumption as module
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_COICOP_DIVISION_COLUMNS,
        DiagnosticUnavailableError,
        desnz_cars_share_by_fuel,
        division_capture_or_reason,
    )

    draws = pd.DataFrame(
        {column: [1.0e3, 2.0e3] for column in UK_LCFS_COICOP_DIVISION_COLUMNS}
    )
    weights = np.array([1.0e7, 5.0e6])
    real = module.vendored_rows

    def doubled(*args, **kwargs):
        rows = list(real(*args, **kwargs))
        return rows + rows

    monkeypatch.setattr(module, "vendored_rows", doubled)
    with pytest.raises(ValueError, match="expected one") as error:
        division_capture_or_reason(draws, weights=weights, period_value=2024)
    assert not isinstance(error.value, DiagnosticUnavailableError)
    with pytest.raises(ValueError, match="duplicate") as error:
        desnz_cars_share_by_fuel(2024)
    assert not isinstance(error.value, DiagnosticUnavailableError)


def test_report_only_diagnostics_record_a_missing_year_and_do_not_raise(
    monkeypatch,
) -> None:
    """A report-only block says why it has no figures; it never stops a build."""

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime import lcfs_consumption as module
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_COICOP_DIVISION_COLUMNS,
        DiagnosticUnavailableError,
        desnz_cars_share_by_fuel,
        division_capture,
        division_capture_or_reason,
        fuel_litres_audit,
    )

    draws = pd.DataFrame(
        {column: [1.0e3, 2.0e3] for column in UK_LCFS_COICOP_DIVISION_COLUMNS}
    )
    weights = np.array([1.0e7, 5.0e6])
    # No Consumer Trends row is vendored for 2031.
    with pytest.raises(DiagnosticUnavailableError, match="row for 2031"):
        division_capture(draws, weights=weights, period_value=2031)
    recorded = division_capture_or_reason(draws, weights=weights, period_value=2031)
    assert recorded["available"] is False and recorded["gated"] is False
    assert "row for 2031" in recorded["reason"]
    assert "divisions" not in recorded
    assert "available" not in division_capture_or_reason(
        draws, weights=weights, period_value=2024
    )

    # DESNZ's vehicle split is vendored for 2022 to 2024 only.
    with pytest.raises(DiagnosticUnavailableError, match="cars row"):
        desnz_cars_share_by_fuel(2031)

    def no_split(period_value: int):
        raise DiagnosticUnavailableError(f"no cars row for {period_value}.")

    monkeypatch.setattr(module, "desnz_cars_share_by_fuel", no_split)
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    fuel = pd.DataFrame(
        {"petrol_spending": [1500.0, 0.0], "diesel_spending": [0.0, 1200.0]}
    )
    audit = fuel_litres_audit(fuel, weights=weights, stage=stage)
    # The audit falls back to the OBR cars share on both fuels and says so.
    assert audit["per_fuel_ratio_basis"] == (
        "uniform cars share, not a per-fuel benchmark"
    )
    assert "no cars row" in audit["cars_share_unavailable"]
    petrol = audit["fuels"]["petrol_spending"]
    assert "desnz_cars_share" not in petrol
    assert petrol["cars_litres_benchmark"] == pytest.approx(
        petrol["hmrc_litres_all_road_users"]
        * audit["obr_fuel_duty_receipts"]["cars_share"]
    )
    assert audit["frame_over_cars_benchmark"] == pytest.approx(
        audit["frame_over_obr_uniform_cars_benchmark"]
    )


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
            "CAL_WEIGHT_ANNUAL": [1.0, 3.0],
            "C72211": [10.0, 0.0],
            "C72212": [0.0, 5.0],
            "C72213": [1.0, 0.0],
        }
    )
    assert lcfs_other_road_fuel_share(donor) == pytest.approx(1.0 / 26.0)
    # A refund netting a diary line below zero is floored, as on the donor.
    refunded = donor.assign(C72213=[1.0, -4.0])
    assert lcfs_other_road_fuel_share(refunded) == pytest.approx(1.0 / 26.0)
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
        "uk_lcfs_2024_25_consumption:road_fuel_total",
        "uk_lcfs_2024_25_consumption:petrol_share_of_road_fuel",
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


#: Donor support wide enough that no recomposed total is held at its maximum.
_WIDE_RECOMPOSE_SUPPORT = {
    "housing_water_and_electricity_consumption": (0.0, 1e9),
    "transport_consumption": (0.0, 1e9),
}


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
            "liquid_fuel_consumption": [300.0, 0.0, 0.0],
            "solid_fuel_consumption": [0.0, 50.0, 0.0],
            "petrol_spending": [800.0, 0.0, 0.0],
            "diesel_spending": [0.0, 700.0, 0.0],
            "bus_fare_spending": [100.0, 0.0, 50.0],
        }
    )
    # The levelled parts, beside the totals as the chain drew them; the
    # heating fuels are drawn parts, added back unchanged.
    draws = pd.DataFrame(
        {
            "housing_water_and_electricity_consumption": [7000.0, 600.0, 800.0],
            "electricity_consumption": [900.0, 600.0, 0.0],
            "gas_consumption": [700.0, 0.0, 0.0],
            "liquid_fuel_consumption": [300.0, 0.0, 0.0],
            "solid_fuel_consumption": [0.0, 50.0, 0.0],
            "transport_consumption": [2000.0, 1000.0, 0.0],
            "petrol_spending": [1500.0, 0.0, 0.0],
            "diesel_spending": [0.0, 900.0, 0.0],
            # Priced bus fares: journeys times yield, not the chain's draw.
            "bus_fare_spending": [0.0, 250.0, 400.0],
            "food_and_non_alcoholic_beverages_consumption": [1.0, 2.0, 3.0],
        }
    )
    weights = np.array([2.0, 1.0, 3.0])

    out, receipt = recompose_parent_totals(
        draws,
        drawn=drawn,
        parameters=_recompose_declaration(),
        weights=weights,
        donor_support=_WIDE_RECOMPOSE_SUPPORT,
    )

    # Housing: 7000 - 2300 + 1900; 600 - 950 floors at 0, + 650, so the total
    # holds its solid fuel as well as its electricity; 800 + 0.
    assert out["housing_water_and_electricity_consumption"].tolist() == [
        6600.0,
        650.0,
        800.0,
    ]
    # Transport nets the drawn fuel and bus and adds the levelled fuel and the
    # priced bus: 2000 - 900 + 1500; 1000 - 700 + 1150; 0 - 50 floors at 0, + 400.
    assert out["transport_consumption"].tolist() == [2600.0, 1450.0, 400.0]
    assert out["food_and_non_alcoholic_beverages_consumption"].tolist() == [
        1.0,
        2.0,
        3.0,
    ]
    # Without uprating factors every part is at the parent's factor: no residue.
    for parent in receipt["parents"].values():
        assert parent["uprating_residue"]["weighted_residue_gbp"] == 0.0

    # With the parent on a 1.02 bridge, drawn energy and bus left at 1.0 and
    # petrol at 0.9, the remainders keep the gap between the factors: housing
    # (2 x 2300 + 950) x 0.02; transport 1600 x (1.02 / 0.9 - 1) + 700 x 0.02
    # + 350 x 0.02 (the #1167 re-review's item 2, recorded not corrected).
    _, with_factors = recompose_parent_totals(
        draws,
        drawn=drawn,
        parameters=_recompose_declaration(),
        weights=weights,
        donor_support=_WIDE_RECOMPOSE_SUPPORT,
        uprating_factors={
            "housing_water_and_electricity_consumption": 1.02,
            "transport_consumption": 1.02,
            "petrol_spending": 0.9,
        },
    )
    housing = with_factors["parents"]["housing_water_and_electricity_consumption"]
    transport = with_factors["parents"]["transport_consumption"]
    assert housing["uprating_residue"]["weighted_residue_gbp"] == pytest.approx(111.0)
    assert transport["uprating_residue"]["weighted_residue_gbp"] == pytest.approx(
        1600.0 * (1.02 / 0.9 - 1.0) + 700.0 * 0.02 + 350.0 * 0.02
    )
    assert transport["uprating_residue"]["part_factors"]["petrol_spending"] == 0.9
    housing = receipt["parents"]["housing_water_and_electricity_consumption"]
    assert housing["drawn_subtracts"] == [
        "domestic_energy_consumption",
        "liquid_fuel_consumption",
        "solid_fuel_consumption",
    ]
    assert housing["weighted_drawn_total"] == 2.0 * 7000.0 + 600.0 + 3.0 * 800.0
    assert housing["rows_floored"] == 1
    assert housing["weighted_floored_mass"] == 350.0
    assert housing["weighted_remainder"] == pytest.approx(
        housing["weighted_drawn_total"]
        - housing["weighted_drawn_subtracts"]
        + housing["weighted_floored_mass"]
    )
    assert housing["rows_below_components"] == 0
    assert housing["maximum_total"] == "donor_support"
    assert housing["rows_held_at_maximum"] == 0
    assert housing["weighted_held_mass"] == 0.0
    transport = receipt["parents"]["transport_consumption"]
    assert transport["drawn_subtracts"] == [
        "petrol_spending",
        "diesel_spending",
        "bus_fare_spending",
    ]
    assert transport["weighted_total"] == 2.0 * 2600.0 + 1450.0 + 3.0 * 400.0
    assert transport["weighted_components"] == 2.0 * 1500.0 + 1150.0 + 3.0 * 400.0
    assert transport["rows_floored"] == 1
    published = receipt["published_fuels"]
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
    assert [entry["value"] for entry in published["classes"]] == expected
    assert [entry["column"] for entry in published["classes"]] == [
        "liquid_fuel_consumption",
        "solid_fuel_consumption",
    ]
    assert published["total"] == pytest.approx(sum(expected))


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
            lambda d: d["published_fuels"].update(resource="family_resources.json"),
            "not a vendored",
        ),
        (
            lambda d: d["published_fuels"]["classes"][0].update(concept="ons.other"),
            "expected one",
        ),
        (
            lambda d: d["published_fuels"]["classes"][0].update(column="gas"),
            "not a heating-fuel column",
        ),
        (
            lambda d: d["published_fuels"]["classes"].pop(),
            "once each",
        ),
        (lambda d: d.pop("maximum_total"), "maximum_total"),
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
        "liquid_fuel_consumption",
        "solid_fuel_consumption",
        "transport_consumption",
        "petrol_spending",
        "diesel_spending",
        "bus_fare_spending",
    )
    draws = pd.DataFrame({column: [1.0] for column in columns})
    with pytest.raises(ValueError, match=match):
        recompose_parent_totals(
            draws,
            drawn=draws,
            parameters=declared,
            weights=[1.0],
            donor_support=_WIDE_RECOMPOSE_SUPPORT,
        )


def test_recompose_holds_each_total_to_the_donor_maximum() -> None:
    """A total is held to the donor's maximum of it, never below its parts (#1164)."""

    from microcosm.build.uk_runtime.lcfs_consumption import recompose_parent_totals

    drawn = pd.DataFrame(
        {
            "domestic_energy_consumption": [2000.0, 100.0, 0.0],
            "liquid_fuel_consumption": [0.0, 0.0, 0.0],
            "solid_fuel_consumption": [0.0, 0.0, 0.0],
            "petrol_spending": [0.0, 0.0, 0.0],
            "diesel_spending": [0.0, 0.0, 0.0],
            "bus_fare_spending": [0.0, 0.0, 0.0],
        }
    )
    draws = pd.DataFrame(
        {
            # Remainders 5800, 4900 and 3000 before the levelled energy.
            "housing_water_and_electricity_consumption": [7800.0, 5000.0, 3000.0],
            "electricity_consumption": [2500.0, 700.0, 4000.0],
            "gas_consumption": [1500.0, 200.0, 3000.0],
            "liquid_fuel_consumption": [0.0, 0.0, 0.0],
            "solid_fuel_consumption": [0.0, 0.0, 0.0],
            "transport_consumption": [10.0, 10.0, 10.0],
            "petrol_spending": [0.0, 0.0, 0.0],
            "diesel_spending": [0.0, 0.0, 0.0],
            "bus_fare_spending": [0.0, 0.0, 0.0],
        }
    )
    support = {**_WIDE_RECOMPOSE_SUPPORT}
    support["housing_water_and_electricity_consumption"] = (0.0, 6500.0)

    out, receipt = recompose_parent_totals(
        draws,
        drawn=drawn,
        parameters=_recompose_declaration(),
        weights=np.array([2.0, 1.0, 3.0]),
        donor_support=support,
    )

    # 5800 + 4000 is held at 6500; 4900 + 900 is within it; 3000 + 7000 would
    # be held at 6500, below its parts, so it keeps the parts alone.
    assert out["housing_water_and_electricity_consumption"].tolist() == [
        6500.0,
        5800.0,
        7000.0,
    ]
    housing = receipt["parents"]["housing_water_and_electricity_consumption"]
    assert housing["rows_held_at_maximum"] == 2
    assert housing["weighted_held_mass"] == 2.0 * 3300.0 + 3.0 * 3000.0
    assert housing["rows_below_components"] == 0
    assert housing["weighted_total"] == pytest.approx(
        housing["weighted_remainder"]
        + housing["weighted_components"]
        - housing["weighted_held_mass"]
    )
    with pytest.raises(ValueError, match="no donor support"):
        recompose_parent_totals(
            draws,
            drawn=drawn,
            parameters=_recompose_declaration(),
            weights=np.array([2.0, 1.0, 3.0]),
            donor_support={"transport_consumption": (0.0, 1e9)},
        )


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
        stage, draws, drawn=draws, weights=[1.0], donor_support={}
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


def test_prepayment_meter_donor_rule_reads_key_and_meter_payments() -> None:
    # microcosm#1110: b1751, b1701 and b233 are what the household last put on an
    # electricity, gas or combined key/meter, c45114/c45214 the diary's
    # slot-meter lines; an account payment (b489) is not prepayment.
    from microcosm.build.uk_runtime.lcfs_consumption import (
        prepayment_meter_from_lcfs,
    )

    household = pd.DataFrame(
        _energy_codes(
            6,
            b1751=[9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            b1701=[0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            b233=[0.0, 0.0, 12.0, 0.0, 0.0, 0.0],
            c45114=[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            c45214=[0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        )
    ).assign(b489=[0.0, 0.0, 0.0, 0.0, 20.0, 0.0])
    assert prepayment_meter_from_lcfs(household).tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    with pytest.raises(ValueError, match="b233"):
        prepayment_meter_from_lcfs(household.drop(columns="b233"))


def _prepayment_donor_and_recipient() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = 400
    tenure = rng.choice(["RENT_FROM_COUNCIL", "RENT_PRIVATELY", "OWNED_OUTRIGHT"], n)
    donor = pd.DataFrame(
        {
            "household_adult_count": rng.integers(1, 4, n),
            "household_child_count": rng.integers(0, 3, n),
            "region": rng.choice(["LONDON", "WALES", "NORTHERN_IRELAND"], n),
            "employment_income": rng.uniform(0, 6e4, n),
            "self_employment_income": rng.uniform(0, 1e4, n),
            "private_pension_income": rng.uniform(0, 2e4, n),
            "hbai_household_net_income": rng.uniform(5e3, 8e4, n),
            "tenure_type": tenure,
            "accommodation_type": rng.choice(["FLAT", "HOUSE_TERRACED"], n),
            "num_vehicles": rng.integers(0, 3, n),
            "household_weight": rng.uniform(500, 1500, n),
        }
    )
    # Renters prepay more often, so the flag depends on the predictors.
    renter = tenure != "OWNED_OUTRIGHT"
    donor["uses_energy_prepayment_meter"] = (
        rng.random(n) < np.where(renter, 0.4, 0.05)
    ).astype(float)
    recipient = donor.drop(
        columns=["household_weight", "uses_energy_prepayment_meter"]
    ).iloc[:120]
    return donor, recipient.reset_index(drop=True)


def _prepayment_parameters(**changes) -> dict:
    return {
        "kind": "impute_prepayment_meter",
        "output": "uses_energy_prepayment_meter",
        "codes": ["b1751", "b1701", "b233", "c45114", "c45214"],
        "rule": "key_or_meter_payment_positive",
        "seed": 0,
        "salt": "lcfs_prepayment_meter",
        "n_estimators": 8,
        **changes,
    }


def test_prepayment_meter_draw_is_keyed_on_the_household() -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import impute_prepayment_meter

    donor, recipient = _prepayment_donor_and_recipient()
    ids = np.arange(1001, 1001 + len(recipient))
    weights = np.full(len(recipient), 250.0)

    first = impute_prepayment_meter(
        donor=donor,
        recipient=recipient,
        household_ids=ids,
        weights=weights,
        parameters=_prepayment_parameters(),
    )
    order = np.random.default_rng(3).permutation(len(recipient))
    permuted = impute_prepayment_meter(
        donor=donor,
        recipient=recipient.iloc[order].reset_index(drop=True),
        household_ids=ids[order],
        weights=weights[order],
        parameters=_prepayment_parameters(),
    )

    assert first.flags.dtype == bool
    assert 0 < first.flags.sum() < len(recipient)
    assert (permuted.flags == first.flags[order]).all()
    receipt = first.receipt
    assert receipt["operation"] == "impute_prepayment_meter"
    assert receipt["donor_share"] == pytest.approx(
        np.average(
            donor["uses_energy_prepayment_meter"], weights=donor["household_weight"]
        )
    )
    assert receipt["recipient_share"] == pytest.approx(first.flags.mean())
    gb = recipient["region"].to_numpy() != "NORTHERN_IRELAND"
    assert receipt["recipient_share_great_britain"] == pytest.approx(
        first.flags[gb].mean()
    )
    assert receipt["recipient_households_great_britain"] == pytest.approx(
        250.0 * first.flags[gb].sum()
    )
    assert first.fit_weight_records[0].fit_name.endswith(
        ":uses_energy_prepayment_meter"
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"output": "prepays"}, "must write"),
        ({"codes": ["b489", "b490"]}, "must read"),
        ({"rule": "b489_or_b490_nonzero"}, "unsupported"),
        ({"salt": ""}, "declared salt"),
        ({"seed": "0"}, "integer seed"),
    ],
)
def test_prepayment_meter_step_refuses_a_different_declaration(
    changes, message
) -> None:
    from microcosm.build.uk_runtime.lcfs_consumption import impute_prepayment_meter

    donor, recipient = _prepayment_donor_and_recipient()
    with pytest.raises(ValueError, match=message):
        impute_prepayment_meter(
            donor=donor,
            recipient=recipient,
            household_ids=np.arange(len(recipient)),
            weights=np.ones(len(recipient)),
            parameters=_prepayment_parameters(**changes),
        )


def test_prepayment_meter_is_exported_for_the_engines_that_read_it() -> None:
    # policyengine-uk reads uses_energy_prepayment_meter from 2.123.3 for the
    # Energy Price Guarantee prepayment discount; an older engine ignores a
    # column it does not know, so the release carries it now (microcosm#1110).
    from microcosm.build.uk_runtime.national_frame import (
        UK_RELEASE_EXPORT_DROPPED_COLUMNS,
        uk_release_export_frame,
    )

    assert (
        "uses_energy_prepayment_meter"
        not in UK_RELEASE_EXPORT_DROPPED_COLUMNS["household"]
    )
    frame = uk_national_frame(
        person=pd.DataFrame(
            {"person_id": [1], "person_benunit_id": [1], "person_household_id": [10]}
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame(
            {
                "household_id": [10],
                "household_weight": [1.0],
                "uses_energy_prepayment_meter": [True],
            }
        ),
        time_period="2024",
    )
    exported = uk_release_export_frame(frame)
    assert exported.table("household")["uses_energy_prepayment_meter"].tolist() == [
        True
    ]


def test_gas_payer_draws_rank_likely_payers_and_draw_positive_gas() -> None:
    """The connection walk's inputs come from two donor gas fits (#1164).

    The classifier's probability of paying for gas follows the donor (here,
    houses pay and flats mostly do not), and the amount model draws a positive
    gas spend for every recipient; a donor without zero-gas households has
    nobody to learn non-payers from and is refused.
    """

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.lcfs_consumption import (
        gas_payer_draws,
        lcfs_energy_pricing,
    )

    energy = lcfs_energy_pricing(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    assert energy is not None and energy.connect_rule == "gas_payer_probability_order"
    rng = np.random.default_rng(4)
    n = 400

    def table(rows: int) -> pd.DataFrame:
        flat = np.arange(rows) % 2 == 0
        return pd.DataFrame(
            {
                "household_adult_count": np.full(rows, 2.0),
                "household_child_count": np.zeros(rows),
                "region": np.full(rows, "LONDON", dtype=object),
                "employment_income": rng.uniform(1e4, 6e4, rows),
                "self_employment_income": np.zeros(rows),
                "private_pension_income": np.zeros(rows),
                "hbai_household_net_income": rng.uniform(1e4, 6e4, rows),
                "tenure_type": np.full(rows, "OWNED_OUTRIGHT", dtype=object),
                "accommodation_type": np.where(flat, "FLAT", "HOUSE_DETACHED"),
                "num_vehicles": np.ones(rows),
            }
        )

    donor = table(n)
    flats = donor["accommodation_type"].eq("FLAT").to_numpy()
    donor["household_weight"] = 1000.0
    donor["gas_consumption"] = np.where(
        flats & (np.arange(n) % 8 != 0), 0.0, rng.uniform(400.0, 1500.0, n)
    )
    for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS:
        if column not in donor:
            donor[column] = 0.0
    recipient = table(60)
    draws = gas_payer_draws(
        donor=donor,
        recipient=recipient,
        household_ids=np.arange(60) + 1,
        energy=energy,
    )

    recipient_flats = recipient["accommodation_type"].eq("FLAT").to_numpy()
    assert draws.probability.shape == (60,)
    assert (
        draws.probability[~recipient_flats].mean()
        > draws.probability[recipient_flats].mean() + 0.3
    )
    assert (draws.positive_spend > 0).all()
    assert draws.receipt["amount_regime"] == "positive_only"
    assert draws.receipt["salt"] == "lcfs_gas_connection"
    assert 0.0 < draws.receipt["donor_gas_payer_share"] < 1.0
    assert [
        record.fit_name.rsplit(":", 1)[1] for record in draws.fit_weight_records
    ] == [
        "gas_payer",
        "gas_payer_amount",
    ]
    assert draws.fit_weight_records[0].weight_kind == "explicit"
    everyone = donor.assign(gas_consumption=rng.uniform(400.0, 1500.0, n))
    with pytest.raises(ValueError, match="no zero-gas or no gas-paying"):
        gas_payer_draws(
            donor=everyone,
            recipient=recipient,
            household_ids=np.arange(60) + 1,
            energy=energy,
        )


def test_lcfs_draws_move_only_for_the_households_whose_inputs_move() -> None:
    """Keyed on household ids: reordering, one changed and one added move no other."""

    from test_support.microcosm_build.uk_keyed_draws import (
        assert_draws_are_keyed_on_ids,
    )

    donor = clean_lcfs_consumption_table(*_synthetic_lcfs_donor())
    recipient = donor.iloc[:10].loc[:, list(UK_LCFS_CONSUMPTION_PREDICTORS)]

    def draw(rows: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
        return impute_lcfs_consumption(
            donor, rows, seed=0, n_estimators=5, record_ids=ids
        ).draws.reset_index(drop=True)

    assert_draws_are_keyed_on_ids(
        draw,
        recipient,
        np.arange(1, len(recipient) + 1),
        perturbed_column="hbai_household_net_income",
        perturbed_value=777.0,
    )
