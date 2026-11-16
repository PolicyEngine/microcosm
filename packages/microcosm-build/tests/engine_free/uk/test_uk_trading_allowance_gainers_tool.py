"""The trading allowance gainers tool reads the stage's claim statuses (microcosm#1439).

A release renumbers ``person_id`` and keeps the build's id in ``person_build_id``.
The stage keys its claimant draw on the build's id, so a measure that draws again
on the release's ids measures a different draw from the one the release carries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.trading_allowance_receipts import (
    NON_CLAIMANT,
    TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT,
    TRADING_ALLOWANCE_STATUS_OUTPUT,
    assign_trading_allowance_claim_status,
)
from tools.measure_uk_trading_allowance_gainers import (
    GAINERS_TOLERANCE,
    TARGET_GAINERS,
    TRADING_ALLOWANCE_CLAIMANT_BANDS,
    expected_gainers_check,
    read_claim_statuses,
)

_PEOPLE = 400


def _people() -> pd.DataFrame:
    """Traders in the two drawn bands, with release ids beside build ids."""

    return pd.DataFrame(
        {
            "person_id": np.arange(1, _PEOPLE + 1, dtype=np.int64),
            "person_build_id": 10_000_000_000 + 7 * np.arange(_PEOPLE, dtype=np.int64),
            "self_employment_income": np.linspace(1_500.0, 8_500.0, _PEOPLE),
            "household_is_spi_synthetic": np.zeros(_PEOPLE, dtype=bool),
        }
    )


def _drawn_on(people: pd.DataFrame, ids: str) -> np.ndarray:
    return assign_trading_allowance_claim_status(
        people[ids].to_numpy(),
        people["self_employment_income"].to_numpy(),
        spi_row=people["household_is_spi_synthetic"].to_numpy(),
    ).status


def test_the_statuses_a_release_stores_are_read_as_they_are() -> None:
    people = _people()
    # No draw at the stage's rates gives every trader NON_CLAIMANT.
    people[TRADING_ALLOWANCE_STATUS_OUTPUT] = NON_CLAIMANT
    people[TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT] = False

    status, full_relief, source = read_claim_statuses(people)

    assert source == "stored"
    assert (status == NON_CLAIMANT).all()
    assert not full_relief.any()


def test_without_stored_statuses_the_draw_is_made_on_the_build_ids() -> None:
    people = _people()

    status, full_relief, source = read_claim_statuses(people)

    assert source == "drawn on person_build_id"
    assert (status == _drawn_on(people, "person_build_id")).all()
    assert not full_relief.any()
    # The draw on the release's renumbered ids is another draw.
    assert (status != _drawn_on(people, "person_id")).any()


def test_a_frame_with_no_build_ids_draws_on_its_own_person_id() -> None:
    people = _people().drop(columns=["person_build_id"])

    status, _, source = read_claim_statuses(people)

    assert source == "drawn on person_id"
    assert (status == _drawn_on(people, "person_id")).all()


@pytest.mark.parametrize(
    "column", [TRADING_ALLOWANCE_STATUS_OUTPUT, TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT]
)
def test_a_stored_column_without_its_partner_is_refused(column: str) -> None:
    people = _people()
    people[column] = (
        NON_CLAIMANT if column == TRADING_ALLOWANCE_STATUS_OUTPUT else False
    )

    with pytest.raises(ValueError, match="without its partner"):
        read_claim_statuses(people)


def _would_be(rows_per_band: tuple[int, int], weight: float = 1_000.0):
    """Would-be gainers in the two drawn bands, one source household each, and a
    person in the undrawn top band and one with no gain."""

    first, second = rows_per_band
    band = np.array([0] * first + [1] * second + [2, 0])
    gain = np.array([300.0] * first + [450.0] * second + [900.0, 0.0])
    weights = np.full(band.shape, weight)
    sources = np.arange(band.size)
    return weights, gain, band, sources


def test_the_check_is_judged_on_each_band_rate_times_its_would_be_gainers() -> None:
    weight, gain, band, sources = _would_be((300, 150))
    rates = [rate for _, rate in TRADING_ALLOWANCE_CLAIMANT_BANDS]

    check = expected_gainers_check(
        weight=weight,
        would_be_gain=gain,
        band=band,
        sources=sources,
        realised={"weighted_gainers": 180_000.0},
    )

    expected = rates[0] * 300_000.0 + rates[1] * 150_000.0
    assert check["judged_on"] == "expected_count"
    assert check["expected_gainers"] == round(expected, -2)
    assert check["expected_mean_gain"] == round(
        (rates[0] * 300_000.0 * 300.0 + rates[1] * 150_000.0 * 450.0) / expected, 0
    )
    assert check["within_tolerance"] == (
        abs(expected / TARGET_GAINERS - 1.0) <= GAINERS_TOLERANCE
    )
    # Equal weights: the Kish effective records are the band's records.
    variance = sum(
        rate * (1.0 - rate) * (rows * 1_000.0) ** 2 / rows
        for rate, rows in zip(rates[:2], (300, 150), strict=True)
    )
    assert check["draw_spread_sd"] == round(float(np.sqrt(variance)), -2)
    assert check["realised_gainers"] == 180_000.0
    assert check["bands"][0]["effective_records"] == 300
    # The undrawn top band adds nothing and shows only its support.
    assert set(check["bands"][2]) == {"upper_inclusive", "rate", "support"}


def test_a_drawn_band_on_fewer_than_three_households_withholds_the_totals() -> None:
    weight, gain, band, sources = _would_be((300, 2))

    check = expected_gainers_check(
        weight=weight,
        would_be_gain=gain,
        band=band,
        sources=sources,
        realised={"weighted_gainers": 180_000.0},
    )

    assert check["withheld"] is True
    assert check["bands"][1]["withheld"] is True
    for total in (
        "expected_gainers",
        "draw_spread_sd",
        "realised_minus_expected_in_sd",
    ):
        assert total not in check
