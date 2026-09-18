"""The NTS bus-travel stage (microcosm#930): cleaning, band draw, trips, eligibility."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.nts_bus_travel import (
    BAND_IDS,
    BUS_IN_LONDON,
    IMPUTE_BUS_USE_BAND_KIND,
    LONDON_GROUP,
    MAX_BAND,
    NON_USER_BAND,
    OTHER_LOCAL_BUS,
    REST_OF_ENGLAND_GROUP,
    SERIES_TRIP_COLUMNS,
    UK_NTS_BUS_TRAVEL_OUTPUT_COLUMNS,
    WEEKS_IN_YEAR,
    NTSBusTravelError,
    NTSColumns,
    UKNTSBusTravelStageTransform,
    assign_bus_pass_eligibility,
    assign_trips_from_band_means,
    band_means_from_donor,
    clean_nts_travel_tables,
    eligibility_rules,
    impute_bus_use_band,
    published_band_shares,
    published_car_availability,
    published_trip_rates,
    recipient_predictors,
)
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate


def _committed_stage():
    return load_country_spec("uk").sources.stage_map()["nts_bus_travel"]


def _operation(stage, kind: str) -> dict:
    return next(dict(op.parameters) for op in stage.operations if op.kind == kind)


def _columns() -> NTSColumns:
    return NTSColumns.from_parameters(
        _operation(_committed_stage(), "clean_nts_travel_tables")
    )


def _synthetic_nts(
    households: int = 90, *, frequency: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = np.arange(households)
    household = pd.DataFrame(
        {
            "HouseholdID": 1000 + rows,
            "SurveyYear": 2022 + rows % 3,
            "HHoldGOR_B02ID": 1 + rows % 9,
            "HHIncome2002_B02ID": 1 + rows % 3,
            "NumCarVan": rows % 4,
            "HHoldNumPeople": 1 + rows % 3,
            "W2": 1.0 + (rows % 5) / 10.0,
        }
    )
    persons = []
    for household_id, size in zip(
        household["HouseholdID"], household["HHoldNumPeople"], strict=True
    ):
        for member in range(int(size)):
            index = int(household_id) * 10 + member
            row = {
                "IndividualID": index,
                "HouseholdID": int(household_id),
                "Age_B01ID": 1 + index % 10,
                "Sex_B01ID": 1 + index % 2,
            }
            if frequency:
                row["LocalBusFreq_B01ID"] = 1 + index % 7
            persons.append(row)
    individual = pd.DataFrame(persons)
    trips = []
    trip_id = 1
    for _, person in individual.iterrows():
        band_code = int(person.get("LocalBusFreq_B01ID", 7))
        count = max(0, 7 - band_code)
        london = (int(person["HouseholdID"]) - 1000) % 9 == 6
        for t in range(count):
            trips.append(
                {
                    "TripID": trip_id,
                    "IndividualID": int(person["IndividualID"]),
                    "HouseholdID": int(person["HouseholdID"]),
                    "MainMode_B04ID": 7 if london else 8,
                    "W5xHH": 1.0 + (t % 3) / 10.0,
                    "JJXSC": 1.0,
                }
            )
            trip_id += 1
        trips.append(
            {
                "TripID": trip_id,
                "IndividualID": int(person["IndividualID"]),
                "HouseholdID": int(person["HouseholdID"]),
                "MainMode_B04ID": 3,
                "W5xHH": 1.0,
                "JJXSC": 1.0,
            }
        )
        trip_id += 1
    return household, individual, pd.DataFrame(trips)


def test_declared_columns_and_codes_parse_from_the_committed_stage() -> None:
    columns = _columns()
    assert columns.london_region_code == 7
    assert set(columns.region_codes.values()) == {
        "NORTH_EAST",
        "NORTH_WEST",
        "YORKSHIRE",
        "EAST_MIDLANDS",
        "WEST_MIDLANDS",
        "EAST_OF_ENGLAND",
        "LONDON",
        "SOUTH_EAST",
        "SOUTH_WEST",
    }
    assert columns.mode_codes == {BUS_IN_LONDON: 7, OTHER_LOCAL_BUS: 8}
    assert columns.frequency_codes[1] == MAX_BAND and columns.frequency_codes[7] == 0
    assert columns.survey_years == (2022, 2023, 2024)
    bad = dict(_operation(_committed_stage(), "clean_nts_travel_tables"))
    bad["codes"] = {**bad["codes"], "main_mode": {"bus_in_london": 7}}
    with pytest.raises(NTSBusTravelError, match="main_mode"):
        NTSColumns.from_parameters(bad)


def test_cleaning_annualises_diary_trips_per_series_and_maps_the_codebook() -> None:
    household, individual, trip = _synthetic_nts()
    donor = clean_nts_travel_tables(household, individual, trip, columns=_columns())
    person = donor.person
    assert donor.frequency_source == "interview_band"
    assert len(person) == len(individual)
    assert set(person["region"]) <= set(_columns().region_codes.values())
    assert set(person["income_band"]) == {1, 2, 3}
    assert person["num_vehicles"].between(0, 5).all()
    # Frequency code 1 (3+ a week) is the top band; code 7 the non-user band.
    codes = individual.set_index("IndividualID")["LocalBusFreq_B01ID"]
    expected = person["individual_id"].map(codes).map(lambda c: 7 - int(c))
    assert (person["local_bus_use_band"].to_numpy() == expected.to_numpy()).all()
    # Trips: 52.14 x sum(W5xHH x JJXSC) over the bus trips only.
    bus = trip[trip["MainMode_B04ID"].isin([7, 8])]
    per_person = (bus["W5xHH"] * bus["JJXSC"]).groupby(bus["IndividualID"]).sum()
    expected_trips = person["individual_id"].map(per_person).fillna(0.0) * WEEKS_IN_YEAR
    assert np.allclose(person["local_bus_trips"], expected_trips)
    london = person["residence_group"] == LONDON_GROUP
    assert (person.loc[london, "other_local_bus_trips"] == 0).all()
    assert (person.loc[~london, "bus_in_london_trips"] == 0).all()
    assert donor.receipt["persons"] == len(person)
    assert donor.receipt["annualisation_weeks"] == WEEKS_IN_YEAR
    # A tab whose declared column is absent is refused with the codebook hint.
    with pytest.raises(NTSBusTravelError, match="derive.columns"):
        clean_nts_travel_tables(
            household.drop(columns=["NumCarVan"]), individual, trip, columns=_columns()
        )


def test_cleaning_falls_back_to_published_shares_without_a_frequency_column() -> None:
    household, individual, trip = _synthetic_nts(frequency=False)
    donor = clean_nts_travel_tables(household, individual, trip, columns=_columns())
    assert donor.frequency_source == "published_shares"
    assert "local_bus_use_band" not in donor.person
    assert donor.receipt["frequency_column"] == "localbusfreq_b01id"


def _recipient_frame(n: int = 240, *, seed: int = 3):
    rng = np.random.default_rng(seed)
    regions = np.array(
        ["LONDON", "SOUTH_EAST", "NORTH_WEST", "SCOTLAND", "WALES", "NORTHERN_IRELAND"]
    )[np.arange(n) % 6]
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1),
            "household_weight": rng.uniform(0.5, 3.0, n),
            "region": regions,
            "num_vehicles": np.arange(n) % 3,
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 2 * n + 1),
            "person_benunit_id": np.repeat(np.arange(1, n + 1), 2),
            "person_household_id": np.repeat(np.arange(1, n + 1), 2),
            "age": rng.integers(1, 90, 2 * n).astype(float),
            "gender": np.array(["MALE", "FEMALE"])[np.arange(2 * n) % 2],
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": np.arange(1, n + 1)}),
        household=household,
        time_period="2024",
    )
    return frame, rng


class _Engine:
    country = "uk"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def variable_metadata(self, name):
        return SimpleNamespace(entity="household")

    def materialize(self, frame, variables, period):
        rows = len(frame.table("household"))
        values = {"household_gross_income": self.rng.uniform(5e3, 9e4, rows)}
        return {name: values[name] for name in variables}


def test_band_draw_is_identity_keyed_rounded_and_within_the_donor_range() -> None:
    household, individual, trip = _synthetic_nts()
    columns = _columns()
    donor = clean_nts_travel_tables(household, individual, trip, columns=columns)
    frame, rng = _recipient_frame()
    band_parameters = _operation(_committed_stage(), IMPUTE_BUS_USE_BAND_KIND)
    recipient = recipient_predictors(
        frame,
        _Engine(rng),
        columns=columns,
        income_band_edges=band_parameters["income_band_edges"],
        region_remap=band_parameters["region_remap"],
    )
    assert set(recipient["region"]) <= set(columns.region_codes.values())
    assert set(recipient["income_band"]) <= {1, 2, 3}
    first = impute_bus_use_band(donor.person, recipient, seed=0, n_estimators=4)
    assert first.band.dtype == np.int64
    assert first.band.min() >= NON_USER_BAND and first.band.max() <= MAX_BAND
    assert first.receipt["regime"] == "zero_inflated_positive"
    assert len(first.fit_weight_records) == 1
    # Permuting the recipients moves nothing: the uniforms are keyed by person id.
    order = rng.permutation(len(recipient))
    second = impute_bus_use_band(
        donor.person,
        recipient.iloc[order].reset_index(drop=True),
        seed=0,
        n_estimators=4,
    )
    assert (second.band == first.band[order]).all()


def test_band_means_and_trip_assignment_split_the_series_by_residence_group() -> None:
    household, individual, trip = _synthetic_nts()
    donor = clean_nts_travel_tables(household, individual, trip, columns=_columns())
    means, receipt = band_means_from_donor(donor.person)
    assert set(means) == {BUS_IN_LONDON, OTHER_LOCAL_BUS}
    assert set(means[BUS_IN_LONDON]) == {LONDON_GROUP, REST_OF_ENGLAND_GROUP}
    # Londoners' non-user band carries no trips; their top band carries some.
    assert means[BUS_IN_LONDON][LONDON_GROUP][NON_USER_BAND] == 0.0
    assert means[BUS_IN_LONDON][LONDON_GROUP][MAX_BAND] > 0.0
    assert means[OTHER_LOCAL_BUS][LONDON_GROUP][MAX_BAND] == 0.0
    assert receipt["source"] == "donor_diary_band_means"
    band = np.array([0, MAX_BAND, 3, MAX_BAND])
    groups = np.array(
        [LONDON_GROUP, LONDON_GROUP, REST_OF_ENGLAND_GROUP, REST_OF_ENGLAND_GROUP]
    )
    trips = assign_trips_from_band_means(band, groups, means)
    assert trips[SERIES_TRIP_COLUMNS[BUS_IN_LONDON]][0] == 0.0
    assert (
        trips[SERIES_TRIP_COLUMNS[BUS_IN_LONDON]][1]
        == means[BUS_IN_LONDON][LONDON_GROUP][MAX_BAND]
    )
    assert (
        trips[SERIES_TRIP_COLUMNS[OTHER_LOCAL_BUS]][3]
        == means[OTHER_LOCAL_BUS][REST_OF_ENGLAND_GROUP][MAX_BAND]
    )
    assert np.allclose(
        trips["local_bus_trips"],
        trips[SERIES_TRIP_COLUMNS[BUS_IN_LONDON]]
        + trips[SERIES_TRIP_COLUMNS[OTHER_LOCAL_BUS]],
    )


def test_eligibility_follows_the_declared_statutory_rules() -> None:
    rules = eligibility_rules(
        _operation(_committed_stage(), "assign_bus_pass_eligibility")
    )
    assert set(rules) == {
        "england_outside_london",
        "london",
        "scotland",
        "wales",
        "northern_ireland",
    }
    ages = np.array([10.0, 17.0, 18.0, 21.0, 22.0, 59.0, 60.0, 65.0, 66.0])
    for region, expected in (
        ("SOUTH_EAST", [False, False, False, False, False, False, False, False, True]),
        ("LONDON", [True, True, False, False, False, False, True, True, True]),
        ("SCOTLAND", [True, True, True, True, False, False, True, True, True]),
        ("WALES", [False, False, False, False, False, False, True, True, True]),
        (
            "NORTHERN_IRELAND",
            [False, False, False, False, False, False, False, True, True],
        ),
    ):
        eligible, receipt = assign_bus_pass_eligibility(
            ages, np.array([region] * len(ages)), rules
        )
        assert eligible.tolist() == expected, region
        assert receipt["eligible_persons"] == sum(expected)
    with pytest.raises(NTSBusTravelError, match="no eligibility rule covers"):
        assign_bus_pass_eligibility(ages[:1], np.array(["ATLANTIS"]), rules)


def test_published_facts_read_the_vendored_rows() -> None:
    parameters = _operation(_committed_stage(), IMPUTE_BUS_USE_BAND_KIND)
    rates = published_trip_rates(parameters)
    assert 5.0 < rates[BUS_IN_LONDON] < 25.0
    assert 15.0 < rates[OTHER_LOCAL_BUS] < 45.0
    cars = published_car_availability(parameters)
    assert set(cars) == {"no_car", "one_car", "two_plus_cars"}
    assert abs(sum(cars.values()) - 1.0) < 0.02
    all_ages, older, receipt = published_band_shares(_committed_stage())
    assert set(all_ages) == set(BAND_IDS) == set(older)
    assert abs(sum(all_ages.values()) - 1.0) < 0.005
    assert 0.4 < 1.0 - all_ages[BAND_IDS[NON_USER_BAND]] < 0.6
    with pytest.raises(NTSBusTravelError, match="trip_rates_resource"):
        published_trip_rates(
            {**parameters, "trip_rates_resource": "orr_rail_facts.json"}
        )


def test_stage_transform_end_to_end_on_synthetic_inputs() -> None:
    committed = _committed_stage()
    operations = tuple(
        SourceOperationSpec(
            kind=op.kind, parameters={**op.parameters, "n_estimators": 4}
        )
        if op.kind == IMPUTE_BUS_USE_BAND_KIND
        else op
        for op in committed.operations
    )
    stage = dataclasses.replace(committed, operations=operations)
    household, individual, trip = _synthetic_nts()
    frame, rng = _recipient_frame()
    transform = UKNTSBusTravelStageTransform(
        stage=stage,
        engine=_Engine(rng),
        nts_household=household,
        nts_individual=individual,
        nts_trip=trip,
    )
    result = transform(frame)
    person = result.table("person")
    out_household = result.table("household")
    assert set(UK_NTS_BUS_TRAVEL_OUTPUT_COLUMNS) <= set(person.columns) | set(
        out_household.columns
    )
    assert person["local_bus_use_band"].between(NON_USER_BAND, MAX_BAND).all()
    assert (person["local_bus_trips"] >= 0).all()
    assert (
        person.loc[person["local_bus_use_band"] == NON_USER_BAND, "local_bus_trips"]
        == 0
    ).all()
    summed = person.groupby("person_household_id")["local_bus_trips"].sum()
    assert np.allclose(
        out_household["household_local_bus_trips"].to_numpy(),
        summed.reindex(out_household["household_id"]).fillna(0.0).to_numpy(),
    )
    # Eligibility follows the area rule on the frame's ages.
    london = (
        person["person_household_id"].map(
            out_household.set_index("household_id")["region"]
        )
        == "LONDON"
    )
    region = person["person_household_id"].map(
        out_household.set_index("household_id")["region"]
    )
    assert (person.loc[london & (person["age"] < 18), "bus_pass_eligible"]).all()
    # Outside London and Scotland no under-60 is eligible (England's rule is
    # State Pension age, Wales 60, Northern Ireland 65); Scotland's under-22s are.
    plain = region.isin(["SOUTH_EAST", "NORTH_WEST", "WALES", "NORTHERN_IRELAND"])
    assert (~person.loc[plain & (person["age"] < 60), "bus_pass_eligible"]).all()
    assert (
        person.loc[(region == "SCOTLAND") & (person["age"] < 22), "bus_pass_eligible"]
    ).all()
    evidence = transform.checkpoint_metadata()["evidence"]
    assert {
        "support_clip",
        "donor",
        "band",
        "incidence",
        "band_means",
        "trip_rates",
        "eligibility",
        "vehicle_shares",
    } <= set(evidence)
    assert evidence["incidence"]["frequency_source"] == "interview_band"
    assert 0.0 < evidence["incidence"]["person_user_share"] < 1.0
    assert evidence["trip_rates"]["published_period_value"] == 2024
    assert set(evidence["vehicle_shares"]["published"]) == {
        "no_car",
        "one_car",
        "two_plus_cars",
    }
    assert (
        evidence["eligibility"]["donor_eligible_trip_share"][OTHER_LOCAL_BUS]
        is not None
    )
    clip = evidence["support_clip"]["columns"]
    assert clip["local_bus_use_band"]["clipped_high_rows"] == 0
    assert clip["local_bus_trips"]["clipped_low_rows"] == 0
    assert len(transform.fit_weight_records) == 1


def test_bus_travel_facts_gate_recomputes_the_published_values() -> None:
    stage = _committed_stage()
    parameters = {
        "stage": "nts_bus_travel",
        "check": "bus_travel_facts",
        "trip_rates_period_value": 2024,
        "maximum_user_share_deviation": 0.05,
        "maximum_trip_rate_deviation": 0.15,
    }
    all_ages, _, _ = published_band_shares(stage)
    rates = published_trip_rates(_operation(stage, IMPUTE_BUS_USE_BAND_KIND))
    user = 1.0 - all_ages[BAND_IDS[NON_USER_BAND]]
    evidence = {
        "stage": "nts_bus_travel",
        "incidence": {
            "frequency_source": "interview_band",
            "person_user_share": user + 0.01,
        },
        "trip_rates": {
            "frame_trips_per_person": {k: v * 1.05 for k, v in rates.items()},
            "published_trips_per_person": dict(rates),
        },
    }
    passed = uk_stage_health_gate(
        evidence=evidence,
        stage="nts_bus_travel",
        check="bus_travel_facts",
        parameters=parameters,
    )
    assert passed.passed, passed.failures
    assert passed.details["user_share"]["published"] == pytest.approx(user)
    failing = {
        "stage": "nts_bus_travel",
        "incidence": {
            "frequency_source": "interview_band",
            "person_user_share": user + 0.2,
        },
        "trip_rates": {
            "frame_trips_per_person": {k: v * 1.5 for k, v in rates.items()},
            "published_trips_per_person": {k: v * 2 for k, v in rates.items()},
        },
    }
    failed = uk_stage_health_gate(
        evidence=failing,
        stage="nts_bus_travel",
        check="bus_travel_facts",
        parameters=parameters,
    )
    assert not failed.passed
    text = " ".join(failed.failures)
    assert "user share" in text and "not the vendored" in text and "above" in text
    with pytest.raises(ValueError, match="trip_rates_period_value"):
        uk_stage_health_gate(
            evidence=evidence,
            stage="nts_bus_travel",
            check="bus_travel_facts",
            parameters={**parameters, "trip_rates_period_value": 2023},
        )
