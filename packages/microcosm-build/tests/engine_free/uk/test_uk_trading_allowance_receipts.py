"""The self-employed's trading allowance claim status (microcosm#1428, #1439)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.receipt_counts import rounded_count, small_cell_share
from microcosm.build.uk_runtime.trading_allowance_receipts import (
    TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT,
    TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME,
    TRADING_ALLOWANCE_STATUS_OUTPUT,
    UKTradingAllowanceReceiptsStageTransform,
    assign_trading_allowance_claim_status,
    claimant_rates,
)


def _stage():
    return load_country_spec("uk").sources.stage_map()[
        TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME
    ]


def _frame(profit: np.ndarray, *, spi=None):
    ids = np.arange(1, len(profit) + 1, dtype=np.int64)
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_source_id": ids,
            "household_weight": 2.0,
            "region": "LONDON",
            "household_is_spi_synthetic": (
                np.zeros(len(ids), dtype=bool) if spi is None else np.asarray(spi)
            ),
        }
    )
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_benunit_id": ids,
            "person_household_id": ids,
            "age": 40,
            "self_employment_income": np.asarray(profit, dtype=np.float64),
        }
    )
    benunit = pd.DataFrame({"benunit_id": ids})
    return uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2024"
    )


def test_claimant_rates_follow_the_profit_bands_with_inclusive_upper_bounds() -> None:
    profit = np.asarray([-500.0, 0.0, 0.01, 4_500.0, 4_500.01, 9_000.0, 9_000.01])
    assert claimant_rates(profit).tolist() == [0.0, 0.0, 0.72, 0.72, 0.14, 0.14, 0.0]


def test_drawn_claimants_claim_and_other_traders_do_not() -> None:
    profit = np.asarray([0.0, -1_200.0, 800.0, 6_000.0, 25_000.0] * 200)
    ids = np.arange(len(profit))
    statuses = assign_trading_allowance_claim_status(
        ids, profit, spi_row=np.zeros(len(profit), dtype=bool)
    )
    trader = profit > 0.0
    claimant = statuses.claimant
    # No positive profit: UNKNOWN, which the engine reads whatever is stored.
    assert (statuses.status[~trader] == "UNKNOWN").all()
    assert not claimant[~trader].any()
    assert (statuses.status[claimant] == "CLAIMANT").all()
    # The full-relief share is zero, so every other trader is NON_CLAIMANT.
    assert not statuses.full_relief.any()
    assert (statuses.status[trader & ~claimant] == "NON_CLAIMANT").all()
    # Nobody above GBP 9,000 claims; the draw is the person's own uniform.
    assert not claimant[profit > 9_000.0].any()
    uniforms = stable_identity_uniforms(
        ids[trader], seed=0, salt="trading_allowance_claimant"
    )
    assert (
        claimant[trader].tolist()
        == (uniforms < claimant_rates(profit[trader])).tolist()
    )
    # Two hundred draws per band land near the band's rate.
    assert claimant[profit == 800.0].mean() == pytest.approx(0.72, abs=0.1)
    assert claimant[profit == 6_000.0].mean() == pytest.approx(0.14, abs=0.1)


def test_the_full_relief_group_holds_only_small_frs_traders_who_do_not_claim() -> None:
    """The group's rule, at a share of one so every member of its population is in
    it: FRS respondents with a profit of GBP 1,000 or less who are not drawn
    claimants. SPI rows never join, nor larger profits (microcosm#1439)."""

    profit = np.asarray([0.0, 400.0, 1_000.0, 1_000.01, 400.0] * 200)
    spi = np.tile([False, False, False, False, True], 200)
    ids = np.arange(len(profit))

    statuses = assign_trading_allowance_claim_status(
        ids, profit, spi_row=spi, full_relief_rate=1.0
    )

    population = (profit > 0.0) & (profit <= 1_000.0) & ~spi & ~statuses.claimant
    np.testing.assert_array_equal(statuses.full_relief, population)
    assert population.any()
    assert (statuses.status[statuses.full_relief] == "UNKNOWN").all()
    assert (
        statuses.status[(profit > 1_000.0) & ~statuses.claimant] == "NON_CLAIMANT"
    ).all()
    assert (statuses.status[spi & ~statuses.claimant] == "NON_CLAIMANT").all()
    # The claimant draw does not move with the group's share.
    baseline = assign_trading_allowance_claim_status(ids, profit, spi_row=spi)
    np.testing.assert_array_equal(statuses.claimant, baseline.claimant)


def test_the_full_relief_draw_is_keyed_on_the_person_with_its_own_salt() -> None:
    profit = np.full(2_000, 500.0)
    ids = np.arange(len(profit))
    spi = np.zeros(len(profit), dtype=bool)

    statuses = assign_trading_allowance_claim_status(
        ids, profit, spi_row=spi, full_relief_rate=0.5
    )

    population = ~statuses.claimant
    uniforms = stable_identity_uniforms(
        ids[population], seed=0, salt="trading_allowance_full_relief"
    )
    assert statuses.full_relief[population].tolist() == (uniforms < 0.5).tolist()
    assert not statuses.full_relief[statuses.claimant].any()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        assign_trading_allowance_claim_status(
            ids, profit, spi_row=spi, full_relief_rate=1.5
        )


def test_the_stage_stores_statuses_deterministically_and_moves_nothing_else() -> None:
    profit = np.tile([0.0, 900.0, 3_000.0, 7_000.0, 40_000.0], 40)
    frame = _frame(profit)
    first = UKTradingAllowanceReceiptsStageTransform(stage=_stage())(frame)
    second = UKTradingAllowanceReceiptsStageTransform(stage=_stage())(frame)
    person = first.table("person")
    pd.testing.assert_series_equal(
        person[TRADING_ALLOWANCE_STATUS_OUTPUT],
        second.table("person")[TRADING_ALLOWANCE_STATUS_OUTPUT],
    )
    assert set(person[TRADING_ALLOWANCE_STATUS_OUTPUT]) <= {
        "CLAIMANT",
        "NON_CLAIMANT",
        "UNKNOWN",
    }
    assert (
        person.loc[profit <= 0.0, TRADING_ALLOWANCE_STATUS_OUTPUT] == "UNKNOWN"
    ).all()
    assert not person[TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT].any()
    # Receipts stay at the engine's zero: the stage no longer writes them.
    assert "self_employment_gross_receipts" not in person.columns
    before = frame.table("person")
    pd.testing.assert_frame_equal(
        person.drop(
            columns=[
                TRADING_ALLOWANCE_STATUS_OUTPUT,
                TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT,
            ]
        ),
        before,
    )
    np.testing.assert_array_equal(
        first.weights_for("household").values, frame.weights_for("household").values
    )
    (record,) = first.mass_log[len(frame.mass_log) :]
    assert record.declared_factor == 1.0
    assert record.old_total == record.new_total


def test_the_receipt_rounds_counts_and_bands_shares() -> None:
    transform = UKTradingAllowanceReceiptsStageTransform(stage=_stage())
    transform(_frame(np.tile([500.0, 6_000.0, 20_000.0], 50)))
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["traders"] == 150
    assert isinstance(evidence["claimants"], int)
    # The group is empty at a share of zero, and says so.
    assert evidence["full_relief"] == {
        "draw": "trading_allowance_full_relief",
        "rate": 0.0,
        "records": 0,
        "support": 0,
    }
    assert [band["rate"] for band in evidence["bands"]] == [0.72, 0.14, 0.0]
    assert all(band["support"] == "30+" for band in evidence["bands"])
    assert evidence["bands"][0]["weighted_traders"] == pytest.approx(100.0)
    # Fewer than ten source households behind a band print the small-cell
    # label, whatever the rounding rules make it (microcosm#1428 review).
    transform(_frame(np.asarray([500.0] * 5)))
    small = transform.checkpoint_metadata()["evidence"]
    assert small["traders"] == rounded_count(5)
    assert small["bands"][0]["weighted_traders"] == small_cell_share(10.0, 5)


def test_the_stage_refuses_drifted_parameters_and_a_second_writer() -> None:
    stage = _stage()
    operation, receipt = stage.operations
    drifted = replace(
        stage,
        operations=(
            replace(
                operation,
                parameters={
                    **dict(operation.parameters),
                    "full_relief": {
                        **dict(operation.parameters["full_relief"]),
                        "rate": 0.5,
                    },
                },
            ),
            receipt,
        ),
    )
    with pytest.raises(ValueError, match="parameters drifted"):
        UKTradingAllowanceReceiptsStageTransform(stage=drifted)(_frame([500.0]))
    frame = UKTradingAllowanceReceiptsStageTransform(stage=stage)(_frame([500.0]))
    with pytest.raises(ValueError, match="only writer"):
        UKTradingAllowanceReceiptsStageTransform(stage=stage)(frame)


def test_the_stage_refuses_a_frame_without_the_spi_flag() -> None:
    frame = _frame([500.0])
    household = frame.table("household").drop(columns=["household_is_spi_synthetic"])
    bare = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period="2024",
        household_weights=frame.weights_for("household").values,
    )
    with pytest.raises(KeyError, match="household_is_spi_synthetic"):
        UKTradingAllowanceReceiptsStageTransform(stage=_stage())(bare)
