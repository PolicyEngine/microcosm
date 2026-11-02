"""The self-employed's gross receipts for the trading allowance (microcosm#1428)."""

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
    TRADING_ALLOWANCE_RECEIPTS_OUTPUT,
    TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME,
    UKTradingAllowanceReceiptsStageTransform,
    assign_trading_allowance_receipts,
    claimant_rates,
)


def _stage():
    return load_country_spec("uk").sources.stage_map()[
        TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME
    ]


def _frame(profit: np.ndarray):
    ids = np.arange(1, len(profit) + 1, dtype=np.int64)
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_source_id": ids,
            "household_weight": 2.0,
            "region": "LONDON",
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


def test_claimants_add_the_allowance_and_other_traders_the_raised_one() -> None:
    profit = np.asarray([0.0, -1_200.0, 800.0, 6_000.0, 25_000.0] * 200)
    ids = np.arange(len(profit))
    receipts, claimant = assign_trading_allowance_receipts(ids, profit)
    trader = profit > 0.0
    # No positive profit: no receipts, which the engine reads as unknown.
    assert (receipts[~trader] == 0.0).all()
    assert not claimant[~trader].any()
    np.testing.assert_allclose(receipts[claimant], profit[claimant] + 1_000.0)
    np.testing.assert_allclose(
        receipts[trader & ~claimant], profit[trader & ~claimant] + 3_000.0
    )
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


def test_the_stage_stores_receipts_deterministically_and_moves_nothing_else() -> None:
    profit = np.tile([0.0, 900.0, 3_000.0, 7_000.0, 40_000.0], 40)
    frame = _frame(profit)
    first = UKTradingAllowanceReceiptsStageTransform(stage=_stage())(frame)
    second = UKTradingAllowanceReceiptsStageTransform(stage=_stage())(frame)
    person = first.table("person")
    pd.testing.assert_series_equal(
        person[TRADING_ALLOWANCE_RECEIPTS_OUTPUT],
        second.table("person")[TRADING_ALLOWANCE_RECEIPTS_OUTPUT],
    )
    expenses = (
        person[TRADING_ALLOWANCE_RECEIPTS_OUTPUT] - person["self_employment_income"]
    )
    assert set(np.round(expenses[profit > 0.0], 6)) <= {1_000.0, 3_000.0}
    assert (person.loc[profit <= 0.0, TRADING_ALLOWANCE_RECEIPTS_OUTPUT] == 0.0).all()
    before = frame.table("person")
    pd.testing.assert_frame_equal(
        person.drop(columns=[TRADING_ALLOWANCE_RECEIPTS_OUTPUT]), before
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
                parameters={**dict(operation.parameters), "claimant_expenses": 900.0},
            ),
            receipt,
        ),
    )
    with pytest.raises(ValueError, match="parameters drifted"):
        UKTradingAllowanceReceiptsStageTransform(stage=drifted)(_frame([500.0]))
    frame = UKTradingAllowanceReceiptsStageTransform(stage=stage)(_frame([500.0]))
    with pytest.raises(ValueError, match="only writer"):
        UKTradingAllowanceReceiptsStageTransform(stage=stage)(frame)
