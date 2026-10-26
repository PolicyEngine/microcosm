"""The gambling participation top-up of the LCFS stage (microcosm#1359), on synthetic facts."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.gambling_participation import (
    NO_ADULTS,
    OCCASIONAL,
    REGULAR,
    UK_GAMBLING_PARTICIPATION_KEYS,
    household_participation,
    household_strata,
    participation_rates,
    participation_settings,
    top_up_gambling_participation,
)
from test_support.microcosm_build.uk_gambling_participation_facts import (
    ACTIVITIES,
    participation_declaration,
    participation_inputs,
    serve_participation_rows,
)

_CELLS = {
    ("MALE", 18): 0.3,
    ("FEMALE", 18): 0.2,
    ("MALE", 35): 0.25,
    ("FEMALE", 35): 0.4,
    ("MALE", 55): 0.1,
    ("FEMALE", 55): 0.15,
}


def _people(*households: tuple[tuple[float, str], ...]) -> pd.DataFrame:
    rows = [
        {"person_household_id": index, "age": age, "gender": sex}
        for index, members in enumerate(households, start=1)
        for age, sex in members
    ]
    return pd.DataFrame(rows)


def test_household_participation_combines_adults_at_theta() -> None:
    person = _people(
        ((20, "MALE"),),
        ((20, "MALE"), (40, "FEMALE"), (10, "FEMALE")),
        ((12, "MALE"),),
        ((20, "MALE"), (40, "FEMALE"), (70, "MALE")),
    )
    ids = [1, 2, 3, 4]

    def chance(theta: float) -> np.ndarray:
        return household_participation(
            person, ids, _CELLS, bands=(18, 35, 55), theta=theta
        )

    # One player per household: the most likely adult's rate.
    assert chance(0.0) == pytest.approx([0.3, 0.4, 0.0, 0.4])
    # Independent adults: any of them plays.
    assert chance(1.0) == pytest.approx([0.3, 1 - 0.6 * 0.7, 0.0, 1 - 0.6 * 0.7 * 0.9])
    assert chance(0.5)[1] == pytest.approx(1 - 0.6 * 0.7**0.5)
    with pytest.raises(ValueError, match="outside"):
        chance(1.5)


def test_household_participation_refuses_an_unknown_sex_or_household() -> None:
    with pytest.raises(ValueError, match="gender outside"):
        household_participation(
            _people(((30, "UNKNOWN"),)), [1], _CELLS, bands=(18, 35, 55), theta=1.0
        )
    with pytest.raises(ValueError, match="does not hold"):
        household_participation(
            _people(((30, "MALE"),)), [2], _CELLS, bands=(18, 35, 55), theta=1.0
        )


def test_household_strata_count_adults_and_band_the_youngest() -> None:
    person = _people(
        ((20, "MALE"),),
        ((40, "MALE"), (60, "FEMALE"), (3, "MALE")),
        ((9, "FEMALE"),),
        ((56, "MALE"), (60, "FEMALE"), (70, "MALE"), (80, "FEMALE")),
    )
    labels = household_strata(
        person, [1, 2, 3, 4], adults_cap=3, youngest_bounds=[35, 55]
    )
    assert list(labels) == [
        "adults_1:youngest_18_to_34",
        "adults_2:youngest_35_to_54",
        NO_ADULTS,
        "adults_3_plus:youngest_55_plus",
    ]


def test_participation_rates_take_each_keys_union_within_its_population() -> None:
    rates = participation_rates(
        participation_declaration(), rows=serve_participation_rows
    )

    # National Lottery products among all gamblers aged 35 to 54 (four-week
    # rate 0.42; men 0.48, twelve months 1.3 times that).
    draws, scratchcards = (
        ACTIVITIES[name][1]
        for name in (
            "national_lottery_draws",
            "national_lottery_scratchcards",
        )
    )
    union = 1 - (1 - draws / 0.42) * (1 - scratchcards / 0.42)
    lottery = rates.rates["national_lottery"]
    assert lottery[REGULAR][("MALE", 35)] == pytest.approx(0.48 * union)
    assert lottery[OCCASIONAL][("MALE", 35)] == pytest.approx(0.48 * union * 1.3)
    # Betting and gaming among non-draw gamblers aged 18 to 34 (0.4; women
    # 0.35), the online-only activity included through its online row.
    product = 1.0
    for name in ("betting", "bingo", "casino_games", "other_online_instant_win_games"):
        product *= 1 - ACTIVITIES[name][0] / 0.4
    betting = rates.rates["betting_and_gaming"]
    assert betting[REGULAR][("FEMALE", 18)] == pytest.approx(0.35 * (1 - product))
    assert betting[OCCASIONAL][("FEMALE", 18)] == pytest.approx(
        0.35 * (1 - product) * 1.5
    )
    # Per band, each key reads its population for all adults and its
    # activities, then both windows for each sex.
    assert rates.receipt["rows_read"] == 3 * (1 + 2 + 4) + 3 * (1 + 4 + 4)
    assert rates.bands == (18, 35, 55)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"resource": "excise_facts.json"}, "must read"),
        ({"age_bands": [[25, "18_to_34"]]}, "rise from 18"),
        ({"windows": {"regular": "past_four_weeks"}}, "windows must map"),
        ({"sexes": {"MALE": "male", "FEMALE": "female"}}, "sexes must map"),
    ],
)
def test_participation_rates_refuse_a_declaration_they_cannot_read(
    change, message
) -> None:
    with pytest.raises(ValueError, match=message):
        participation_rates(
            participation_declaration(**change), rows=serve_participation_rows
        )


def _activities(key: str, **rows) -> dict:
    keys = copy.deepcopy(participation_declaration()["keys"])
    keys[key]["activities"] = rows
    return {"keys": keys}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_activities("betting_and_gaming", poker={}), "expected one .* found 0"),
        # An online-only activity has no all-channels row.
        (
            _activities(
                "betting_and_gaming",
                other_online_instant_win_games={"gambling_channel": "all"},
            ),
            "other_online_instant_win_games.* found 0",
        ),
        # Any gambling's online row in the activity table matches the
        # population table's row without its worksheet ...
        ({"population_dimensions": {}}, "any_gambling.* found 2"),
        # ... and an activity's online row its all-channels row without the
        # channel.
        (
            _activities("national_lottery", national_lottery_draws={}),
            "national_lottery_draws.* found 2",
        ),
        ({"population_dimensions": {"regular": {"sex": "all"}}}, "may not fix"),
        (
            _activities("national_lottery", national_lottery_draws={"sex": "all"}),
            "no activity may fix",
        ),
        (
            {"activity_dimensions": {"gambling_channel": "all"}},
            "must map the classes",
        ),
        (
            {
                "keys": {
                    **participation_declaration()["keys"],
                    "national_lottery": {
                        "population": "any_gambling",
                        "activities": ["national_lottery_draws"],
                        "theta": 0.9,
                    },
                }
            },
            "map each activity",
        ),
    ],
)
def test_participation_rates_read_each_cell_from_its_own_row(change, message) -> None:
    with pytest.raises(ValueError, match=message):
        participation_rates(
            participation_declaration(**change), rows=serve_participation_rows
        )


def _keys(**changes) -> dict:
    keys = copy.deepcopy(participation_declaration()["keys"])
    for key, entry in changes.items():
        if entry is None:
            keys.pop(key)
        else:
            keys[key].update(entry)
    return keys


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"keys": _keys(betting_and_gaming=None)}, "keys must be"),
        ({"keys": _keys(national_lottery={"theta": 1.5})}, "national_lottery theta"),
        ({"occasional_stake_share": 0.0}, "above zero"),
        ({"occasional_stake_share": 1.5}, "occasional_stake_share"),
        ({"strata": {"adults_cap": 0, "youngest_adult_bounds": [35]}}, "adults_cap"),
        (
            {"strata": {"adults_cap": 3, "youngest_adult_bounds": [55, 35]}},
            "youngest_adult_bounds",
        ),
        ({"propensity": {"max_leaf_nodes": 15}}, "propensity must set"),
        ({"salt": ""}, "must declare salt"),
    ],
)
def test_participation_settings_refuse_a_declaration_out_of_range(
    change, message
) -> None:
    with pytest.raises(ValueError, match=message):
        participation_settings(participation_declaration(**change))


def _strata_and_targets(inputs: dict) -> tuple[np.ndarray, dict]:
    """The strata and each key's household chances, recomputed from the facts."""

    settings = participation_settings(inputs["parameters"])
    rates = participation_rates(inputs["parameters"], rows=serve_participation_rows)
    strata = household_strata(
        inputs["person"],
        inputs["household_ids"],
        adults_cap=settings["strata"]["adults_cap"],
        youngest_bounds=settings["strata"]["youngest_adult_bounds"],
    )
    chances = {
        key: {
            window: household_participation(
                inputs["person"],
                inputs["household_ids"],
                rates.rates[key][window],
                bands=rates.bands,
                theta=settings["theta"][key],
            )
            for window in (REGULAR, OCCASIONAL)
        }
        for key in UK_GAMBLING_PARTICIPATION_KEYS
    }
    return strata, chances


def test_top_up_meets_each_stratum_share_within_one_household_weight() -> None:
    inputs = participation_inputs()
    result = top_up_gambling_participation(**inputs)
    strata, chances = _strata_and_targets(inputs)
    weight = inputs["weights"]
    for key in UK_GAMBLING_PARTICIPATION_KEYS:
        drawn = inputs["keys"][key]
        topped = result.keys[key]
        recorded = result.receipt["keys"][key]["strata"]
        for label in set(strata):
            member = strata == label
            mass = weight[member].sum()
            target = np.average(
                chances[key][OCCASIONAL][member], weights=weight[member]
            )
            before = weight[member & (drawn > 0)].sum() / mass
            after = weight[member & (topped > 0)].sum() / mass
            # The walk stops within the weight of one household in the stratum.
            assert abs(after - max(target, before)) * mass <= weight[member].max()
            if recorded[label] != {"withheld": True}:
                regular = np.average(
                    chances[key][REGULAR][member], weights=weight[member]
                )
                assert recorded[label]["regular_target"] == pytest.approx(regular)
                assert recorded[label]["annual_target"] == pytest.approx(target)
                assert recorded[label]["after_annual"] == pytest.approx(after)
                assert (
                    abs(recorded[label]["after_regular"] - regular) * mass
                    <= weight[member].max()
                )


def test_top_up_keeps_the_diarys_stakes_and_stakes_every_added_household() -> None:
    inputs = participation_inputs()
    result = top_up_gambling_participation(**inputs)
    for key in UK_GAMBLING_PARTICIPATION_KEYS:
        drawn, topped = inputs["keys"][key], result.keys[key]
        seen = drawn > 0
        np.testing.assert_array_equal(topped[seen], drawn[seen])
        added = (topped > 0) & ~seen
        assert added.sum() > seen.sum()
        entry = result.receipt["keys"][key]
        assert entry["added_without_stake"] == 0
        assert sum(entry["stake_shares"].values()) == pytest.approx(1.0)
        assert entry["after_annual"] == float(
            inputs["weights"][topped > 0].sum() / inputs["weights"].sum()
        )
    assert [record.fit_name for record in result.fit_weight_records] == [
        "test:national_lottery_participation",
        "test:national_lottery_participation_stake",
        "test:betting_and_gaming_participation",
        "test:betting_and_gaming_participation_stake",
    ]


def test_occasional_players_take_the_declared_share_of_a_drawn_stake() -> None:
    inputs = participation_inputs()
    low = top_up_gambling_participation(**inputs)
    high = top_up_gambling_participation(
        **participation_inputs(
            declaration=participation_declaration(occasional_stake_share=0.2)
        )
    )
    weight = inputs["weights"]
    for key in UK_GAMBLING_PARTICIPATION_KEYS:
        moved = low.keys[key] != high.keys[key]
        assert moved.sum() > 0
        # Only the occasional players move, and by exactly the ratio of shares.
        np.testing.assert_allclose(high.keys[key][moved] / low.keys[key][moved], 2.0)
        share = low.receipt["keys"][key]["stake_shares"]["added_occasional"]
        assert 0 < share < low.receipt["keys"][key]["stake_shares"]["added_regular"]
        # The added regular players' mean stake is the seen households'.
        drawn, topped = inputs["keys"][key], low.keys[key]
        seen = drawn > 0
        regular = (topped > 0) & ~seen & ~moved
        assert np.average(topped[regular], weights=weight[regular]) == pytest.approx(
            np.average(drawn[seen], weights=weight[seen])
        )
        # The receipt carries the scale's terms, so a moved scale shows which
        # side moved (Vahid's review of #1400, round 3).
        entry = low.receipt["keys"][key]
        terms = entry["amount_scale_terms"]
        assert terms["anchor"] == "seen_households"
        assert terms["anchor_mean_stake"] == pytest.approx(
            np.average(drawn[seen], weights=weight[seen])
        )
        assert entry["amount_scale"] == pytest.approx(
            terms["anchor_mean_stake"] / terms["added_regular_drawn_mean_stake"]
        )
        assert terms["seen_drawn_mean_stake"] > 0


def test_top_up_is_keyed_on_household_identity() -> None:
    inputs = participation_inputs()
    result = top_up_gambling_participation(**inputs)
    order = np.random.default_rng(3).permutation(len(inputs["household_ids"]))
    shuffled = {
        **inputs,
        "keys": {key: values[order] for key, values in inputs["keys"].items()},
        "household_ids": inputs["household_ids"][order],
        "weights": inputs["weights"][order],
        "recipient_features": inputs["recipient_features"].iloc[order],
    }
    again = top_up_gambling_participation(**shuffled)
    for key in UK_GAMBLING_PARTICIPATION_KEYS:
        np.testing.assert_allclose(again.keys[key], result.keys[key][order], rtol=1e-12)


def test_a_stratum_already_past_its_shares_adds_no_household() -> None:
    inputs = participation_inputs()
    strata, _ = _strata_and_targets(inputs)
    full = strata == "adults_1:youngest_55_plus"
    for key in UK_GAMBLING_PARTICIPATION_KEYS:
        inputs["keys"][key] = np.where(full, 5.0, inputs["keys"][key])
    result = top_up_gambling_participation(**inputs)
    for key in UK_GAMBLING_PARTICIPATION_KEYS:
        np.testing.assert_array_equal(result.keys[key][full], 5.0)
        entry = result.receipt["keys"][key]["strata"]["adults_1:youngest_55_plus"]
        assert entry["before"] == entry["after_regular"] == entry["after_annual"] == 1.0


def test_the_receipt_withholds_a_stratum_resting_on_few_households() -> None:
    inputs = participation_inputs()
    strata, _ = _strata_and_targets(inputs)
    # Every household in the stratum copies one source household.
    sources = np.asarray(inputs["household_ids"]).copy()
    thin = strata == "adults_2:youngest_18_to_34"
    sources[thin] = sources[thin][0]
    result = top_up_gambling_participation(**{**inputs, "sources": sources})
    for key in UK_GAMBLING_PARTICIPATION_KEYS:
        recorded = result.receipt["keys"][key]["strata"]
        assert recorded["adults_2:youngest_18_to_34"] == {"withheld": True}
        assert any(entry != {"withheld": True} for entry in recorded.values())
        # The national figures rest on every other source household, and the
        # receipt gives no stratum's households, so they give none back.
        assert "withheld" not in result.receipt["keys"][key]
        assert "households" not in recorded["adults_1:youngest_35_to_54"]


def test_top_up_refuses_a_donor_with_too_few_stakes() -> None:
    inputs = participation_inputs()
    stakes = np.zeros(len(inputs["donor_weights"]))
    stakes[:5] = 10.0
    inputs["donor_keys"] = {**inputs["donor_keys"], "national_lottery": stakes}
    with pytest.raises(ValueError, match="10 or more households"):
        top_up_gambling_participation(**inputs)
