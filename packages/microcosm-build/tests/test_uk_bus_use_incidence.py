"""Local-bus use incidence from vendored NTS frequency-of-use shares (#890 I)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.bus_use_incidence import (
    BusUseBandShares,
    assign_bus_use_incidence,
    incidence_operation,
    nts_band_shares,
    under_threshold_shares,
)
from microcosm.build.uk_runtime.lcfs_consumption import lcfs_bus_use_incidence
from microcosm.build.uk_runtime.national_frame import uk_national_frame


def _declared() -> dict:
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    parameters = incidence_operation(stage)
    assert parameters is not None
    return parameters


def test_declared_bands_read_both_nts_series_by_label() -> None:
    shares, receipt = nts_band_shares(_declared())

    assert shares.band_ids[-1] == shares.non_user_band
    assert shares.trips_per_year[shares.non_user_band] == 0.0
    assert set(shares.all_ages) == set(shares.band_ids) == set(shares.older)
    assert sum(shares.all_ages.values()) == pytest.approx(1.0, abs=2e-3)
    assert sum(shares.older.values()) == pytest.approx(1.0, abs=2e-3)
    # NTS0313 2024: 49.511 % use a local bus less than once a year or never;
    # NTS0621 2024: 49.957 % of people aged 60 and over.
    assert receipt["all_ages_user_share"] == pytest.approx(1 - 0.49511, abs=1e-4)
    assert receipt["older_user_share"] == pytest.approx(1 - 0.49957, abs=1e-4)
    assert shares.all_ages["three_or_more_times_a_week"] == pytest.approx(
        0.13512, abs=1e-4
    )
    assert shares.older["three_or_more_times_a_week"] == pytest.approx(
        0.12345, abs=1e-4
    )
    assert receipt["period_value"] == 2024
    assert receipt["age_threshold"] == 60
    assert len(receipt["source_record_ids"]) == 14
    assert receipt["user_definition"] == "at_least_once_a_year"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda p: p.update(resource="need_energy_facts.json"),
            "nts_bus_use_frequency",
        ),
        (lambda p: p["bands"][2]["labels"].pop(), "is not declared"),
        (lambda p: p["bands"][0]["labels"].append("Once or twice a week (%)"), "twice"),
        (lambda p: p.update(non_user_band="never"), "not a declared band"),
        (lambda p: p["bands"][-1].update(trips_per_year=3), "zero trips"),
        (lambda p: p.update(period_value=1999), "missing an NTS series"),
        (lambda p: p.update(user_definition="ever"), "user_definition"),
    ],
)
def test_declaration_refusals(mutation, match: str) -> None:
    import copy

    parameters = copy.deepcopy(_declared())
    parameters["bands"] = [
        dict(band, labels=list(band["labels"])) for band in parameters["bands"]
    ]
    mutation(parameters)
    with pytest.raises(ValueError, match=match):
        nts_band_shares(parameters)


def _toy_shares() -> BusUseBandShares:
    return BusUseBandShares(
        band_ids=("weekly", "yearly", "never"),
        trips_per_year={"weekly": 52.0, "yearly": 1.0, "never": 0.0},
        all_ages={"weekly": 0.3, "yearly": 0.2, "never": 0.5},
        older={"weekly": 0.4, "yearly": 0.2, "never": 0.4},
        non_user_band="never",
        older_age_band="60 and over",
        age_threshold=60,
    )


def test_under_threshold_shares_solve_the_two_published_series() -> None:
    under = under_threshold_shares(_toy_shares(), older_population_share=0.25)
    # p_all = 0.25 * p_older + 0.75 * p_under
    assert under["weekly"] == pytest.approx((0.3 - 0.25 * 0.4) / 0.75)
    assert under["never"] == pytest.approx((0.5 - 0.25 * 0.4) / 0.75)
    assert sum(under.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="inconsistent"):
        under_threshold_shares(_toy_shares(), older_population_share=0.9)
    with pytest.raises(ValueError, match="older population share"):
        under_threshold_shares(_toy_shares(), older_population_share=1.0)


def test_household_incidence_is_identity_keyed_and_rolls_up_any_user() -> None:
    rng = np.random.default_rng(11)
    n_households = 400
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 2 * n_households + 1),
            "person_household_id": np.repeat(np.arange(1, n_households + 1), 2),
            "age": rng.integers(0, 95, 2 * n_households),
        }
    )
    household = pd.DataFrame({"household_id": np.arange(1, n_households + 1)})
    weights = rng.uniform(0.5, 2.0, n_households)

    first = assign_bus_use_incidence(
        person, household, household_weights=weights, shares=_toy_shares(), seed=0
    )
    second = assign_bus_use_incidence(
        person, household, household_weights=weights, shares=_toy_shares(), seed=0
    )
    other_seed = assign_bus_use_incidence(
        person, household, household_weights=weights, shares=_toy_shares(), seed=1
    )

    assert first.household_user.tolist() == second.household_user.tolist()
    assert first.household_user.tolist() != other_seed.household_user.tolist()
    # Household user flag = any member in a user band; trips = members' sum.
    user_by_household = (
        pd.Series(first.person_band != "never")
        .groupby(person["person_household_id"].to_numpy())
        .any()
    )
    assert first.household_user.tolist() == user_by_household.tolist()
    trips = (
        pd.Series([_toy_shares().trips_per_year[band] for band in first.person_band])
        .groupby(person["person_household_id"].to_numpy())
        .sum()
    )
    assert first.household_trips_per_year.tolist() == trips.tolist()
    receipt = first.receipt
    assert receipt["seed"] == 0 and receipt["salt"] == "lcfs_uses_local_bus"
    assert 0.3 < receipt["older_population_share"] < 0.45
    # Roughly half of people use a bus in the toy shares; households more.
    assert 0.4 < receipt["person_user_share"] < 0.6
    assert receipt["household_user_share"] > receipt["person_user_share"]
    assert set(receipt["person_band_shares"]) == {"weekly", "yearly", "never"}
    assert receipt["mean_trips_per_person"] > 0
    # Reordering the persons does not change any household's flag.
    shuffled = person.sample(frac=1.0, random_state=3).reset_index(drop=True)
    again = assign_bus_use_incidence(
        shuffled, household, household_weights=weights, shares=_toy_shares(), seed=0
    )
    assert again.household_user.tolist() == first.household_user.tolist()


def test_stage_helper_draws_from_the_declaration_on_a_frame() -> None:
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "person_benunit_id": [1, 1, 2, 3],
                "person_household_id": [10, 10, 20, 30],
                "age": [70.0, 30.0, 65.0, 8.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2, 3]}),
        household=pd.DataFrame(
            {"household_id": [10, 20, 30], "household_weight": [1.0, 2.0, 3.0]}
        ),
        time_period="2024",
    )
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]

    result = lcfs_bus_use_incidence(stage, frame)

    assert result is not None
    assert len(result.household_user) == 3
    assert result.receipt["nts"]["period_value"] == 2024
    assert result.receipt["older_population_share"] == pytest.approx(3.0 / 7.0)
    assert set(result.receipt["under_threshold_shares"]) == set(
        result.receipt["nts"]["all_ages_shares"]
    )

    class Stage:
        operations = ()

    assert lcfs_bus_use_incidence(Stage(), frame) is None
