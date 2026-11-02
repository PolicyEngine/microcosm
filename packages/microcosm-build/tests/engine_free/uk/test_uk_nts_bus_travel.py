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
    BOARDING_CLASSES,
    BOARDING_CONCESSIONARY,
    BOARDING_FARE_PAID,
    BOARDING_PASS,
    BUS_IN_LONDON,
    IMPUTE_BUS_USE_BAND_KIND,
    LONDON_GROUP,
    MAX_BAND,
    NON_USER_BAND,
    NON_USER_BAND_RULE,
    OTHER_LOCAL_BUS,
    REST_OF_ENGLAND_GROUP,
    SERIES_TRIP_COLUMNS,
    UK_NTS_BUS_TRAVEL_OUTPUT_COLUMNS,
    UK_NTS_PERSON_OUTPUT_COLUMNS,
    UK_NTS_RECEIPT_SUPPRESSED,
    WEEKS_IN_YEAR,
    FareCapRegime,
    NTSBusTravelError,
    NTSColumns,
    UKNTSBusTravelStageTransform,
    assign_bus_pass_eligibility,
    assign_trips_from_band_means,
    band_means_from_donor,
    boarding_class_column,
    clean_nts_travel_tables,
    eligibility_rules,
    impute_bus_use_band,
    published_band_shares,
    published_car_availability,
    published_trip_rates,
    recipient_predictors,
    single_fare_shares_from_donor,
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
    household, individual, trip, _stage, _ticket = _synthetic_nts_with_tickets(
        households, frequency=frequency
    )
    return household, individual, trip


def _synthetic_nts_with_tickets(
    households: int = 90, *, frequency: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = np.arange(households)
    household = pd.DataFrame(
        {
            "HouseholdID": 1000 + rows,
            # Every region in every survey year, so the cap years hold
            # London and rest-of-England donors alike.
            "SurveyYear": 2022 + (rows // 9) % 3,
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
                "Age_B01ID": 1 + index % 21,
                "Sex_B01ID": 1 + index % 2,
            }
            if frequency:
                # The ten OrdBus2Freq_B01ID codes, every one populated.
                row["OrdBus2Freq_B01ID"] = 1 + (int(household_id) + member) % 10
            persons.append(row)
    individual = pd.DataFrame(persons)
    w2 = dict(zip(household["HouseholdID"], household["W2"], strict=True))
    trips = []
    stages = []
    tickets = []
    trip_id = 1
    stage_id = 1
    ticket_id = 1
    for _, person in individual.iterrows():
        code = int(person.get("OrdBus2Freq_B01ID", 10))
        count = max(0, 9 - code)
        london = (int(person["HouseholdID"]) - 1000) % 9 == 6
        household_weight = float(w2[int(person["HouseholdID"])])
        pid = int(person["IndividualID"])
        # Odd-numbered households hold a ticket, alternately a concessionary
        # pass (7) and a season ticket (1); the rest pay at the point of use,
        # with a free boarding every fifth trip.
        held = None
        hid = int(person["HouseholdID"])
        if hid % 2 == 1:
            held = ticket_id
            tickets.append(
                {
                    "IndTicketID": ticket_id,
                    "IndividualID": pid,
                    "SpecialTicket_B01ID": 7 if (hid // 2) % 2 == 0 else 1,
                    "SurveyYear": 2024,
                }
            )
            ticket_id += 1
        for t in range(count):
            trips.append(
                {
                    "TripID": trip_id,
                    "IndividualID": pid,
                    "HouseholdID": int(person["HouseholdID"]),
                    "MainMode_B04ID": 7 if london else 8,
                    # W5 carries W2 (the household_and_trip basis).
                    "W5": household_weight * (1.0 + (t % 3) / 10.0),
                    "JJXSC": 1.0,
                }
            )
            free = held is None and t % 5 == 4
            stages.append(
                {
                    "StageID": stage_id,
                    "TripID": trip_id,
                    "IndividualID": pid,
                    "HouseholdID": int(person["HouseholdID"]),
                    "IndTicketID": held if held is not None else np.nan,
                    "StageMode_B04ID": 7 if london else 8,
                    "NumBoardings": 1 + (t % 4 == 3),
                    "StageFareCost": (
                        0.0 if held is not None or free else 2.0 - (t % 3) * 0.25
                    ),
                    "SurveyYear": 2024,
                }
            )
            stage_id += 1
            trip_id += 1
        trips.append(
            {
                "TripID": trip_id,
                "IndividualID": int(person["IndividualID"]),
                "HouseholdID": int(person["HouseholdID"]),
                "MainMode_B04ID": 3,
                "W5": household_weight,
                "JJXSC": 1.0,
            }
        )
        trip_id += 1
    return (
        household,
        individual,
        pd.DataFrame(trips),
        pd.DataFrame(stages),
        pd.DataFrame(tickets),
    )


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
    # OrdBus2Freq_B01ID: codes 1-3 are the top band, 9 and 10 the non-user
    # band, and -9 (not applicable) is declared a non-user.
    assert columns.frequency_codes[1] == MAX_BAND
    assert columns.frequency_codes[3] == MAX_BAND
    assert columns.frequency_codes[10] == 0 and columns.frequency_codes[-9] == 0
    assert -8 not in columns.frequency_codes
    assert columns.trip_weight_basis == "household_and_trip"
    assert columns.age_band_codes[1] == 0 and columns.age_band_codes[21] == 85
    assert columns.survey_years == (2022, 2023, 2024)
    bad = dict(_operation(_committed_stage(), "clean_nts_travel_tables"))
    bad["codes"] = {**bad["codes"], "main_mode": {"bus_in_london": 7}}
    with pytest.raises(NTSBusTravelError, match="main_mode"):
        NTSColumns.from_parameters(bad)
    # England's £2 cap covers 2023 and 2024 whole; 2022 is before it.
    derive = _operation(_committed_stage(), "clean_nts_travel_tables")
    regime = FareCapRegime.from_parameters(derive, survey_years=columns.survey_years)
    assert regime.cap_fare == 2.0 and regime.cap_tolerance == 0.01
    assert regime.cap_years == (2023, 2024) and regime.pre_cap_years == (2022,)
    stray = {**derive["fare_cap_regime"], "cap_years": [2023, 2025]}
    with pytest.raises(NTSBusTravelError, match="cap_years"):
        FareCapRegime.from_parameters(
            {"fare_cap_regime": stray}, survey_years=columns.survey_years
        )
    overlap = {**derive["fare_cap_regime"], "pre_cap_years": [2022, 2023]}
    with pytest.raises(NTSBusTravelError, match="pre_cap_years"):
        FareCapRegime.from_parameters(
            {"fare_cap_regime": overlap}, survey_years=columns.survey_years
        )


def test_cleaning_annualises_diary_trips_per_series_and_maps_the_codebook() -> None:
    household, individual, trip = _synthetic_nts()
    donor = clean_nts_travel_tables(household, individual, trip, columns=_columns())
    person = donor.person
    assert donor.frequency_source == "interview_band"
    assert len(person) == len(individual)
    assert set(person["region"]) <= set(_columns().region_codes.values())
    assert set(person["income_band"]) == {1, 2, 3}
    assert person["num_vehicles"].between(0, 5).all()
    # The interview code maps onto the band through the declared codebook.
    codes = individual.set_index("IndividualID")["OrdBus2Freq_B01ID"]
    expected = person["individual_id"].map(codes).map(_columns().frequency_codes)
    assert (person["local_bus_use_band"].to_numpy() == expected.to_numpy()).all()
    # Age: the NTS band code's lower age, banded on the declared edges (code 4
    # is 5-10, the second declared band; code 17 is 65-69, the ninth).
    age_codes = individual.set_index("IndividualID")["Age_B01ID"]
    lower = person["individual_id"].map(age_codes).map(_columns().age_band_codes)
    assert (person["age_lower"].to_numpy() == lower.to_numpy()).all()
    assert (
        person.loc[lower == 5, "age_band"].eq(1).all()
        and person.loc[lower == 65, "age_band"].eq(8).all()
    )
    # Trips: 52.14 x sum(W5 x JJXSC) / W2 over the bus trips only (W5 carries W2).
    bus = trip[trip["MainMode_B04ID"].isin([7, 8])]
    w2 = household.set_index("HouseholdID")["W2"]
    counted = bus["W5"] * bus["JJXSC"] / bus["HouseholdID"].map(w2)
    per_person = counted.groupby(bus["IndividualID"]).sum()
    expected_trips = person["individual_id"].map(per_person).fillna(0.0) * WEEKS_IN_YEAR
    assert np.allclose(person["local_bus_trips"], expected_trips)
    assert donor.receipt["trip_weight_basis"] == "household_and_trip"
    assert donor.receipt["households_zero_weight"] == 0
    assert donor.receipt["persons_unmapped_frequency"] == 0
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


def test_stage_and_ticket_tables_class_boardings_and_give_the_single_fare_share() -> (
    None
):
    """Fare paid at the point of use, season tickets, concessionary passes, free."""

    household, individual, trip, stage, ticket = _synthetic_nts_with_tickets()
    donor = clean_nts_travel_tables(
        household, individual, trip, stage, ticket, columns=_columns()
    )
    person = donor.person
    assert donor.receipt["ticket_tables_read"] is True
    classes = donor.receipt["boarding_classes"]
    assert set(classes["by_series"]) == {BUS_IN_LONDON, OTHER_LOCAL_BUS}
    for series in (BUS_IN_LONDON, OTHER_LOCAL_BUS):
        share = classes["by_series"][series]["single_share_of_fare_paying_boardings"]
        assert 0.0 < share < 1.0
        assert classes["by_series"][series]["boardings_per_bus_trip"] >= 1.0
    cell = next(iter(classes["by_group_and_year"].values()))
    assert set(cell["boarding_class_shares"]) == set(BOARDING_CLASSES)
    assert abs(sum(cell["boarding_class_shares"].values()) - 1.0) < 1e-9
    assert 1.0 <= cell["mean_fare_paid_per_boarding"] <= 2.0
    # Per person: a ticket holder has no fare-paid boardings; a payer has no
    # pass boardings; a concessionary holder's boardings are concessionary.
    ticket_of = ticket.set_index("IndividualID")["SpecialTicket_B01ID"]
    for series in (BUS_IN_LONDON, OTHER_LOCAL_BUS):
        paid = person[boarding_class_column(series, BOARDING_FARE_PAID)]
        passes = person[boarding_class_column(series, BOARDING_PASS)]
        conc = person[boarding_class_column(series, BOARDING_CONCESSIONARY)]
        holder = person["individual_id"].map(ticket_of)
        assert (paid[holder.notna()] == 0).all()
        assert (passes[holder.isna()] == 0).all() and (conc[holder.isna()] == 0).all()
        assert (conc[holder == 7] >= 0).all() and (passes[holder == 7] == 0).all()
    shares, receipt = single_fare_shares_from_donor(person)
    assert set(shares) == {BUS_IN_LONDON, OTHER_LOCAL_BUS}
    for series in shares:
        for group in shares[series]:
            assert all(0.0 <= v <= 1.0 for v in shares[series][group].values())
    assert set(receipt["all_england_single_share"]) == {BUS_IN_LONDON, OTHER_LOCAL_BUS}
    trips = assign_trips_from_band_means(
        person["local_bus_use_band"].to_numpy(),
        person["residence_group"].to_numpy(),
        band_means_from_donor(person)[0],
        single_fare_shares=shares,
    )
    share_column = trips["local_bus_single_fare_share"]
    assert share_column.shape == (len(person),)
    assert ((share_column >= 0.0) & (share_column <= 1.0)).all()
    # Without the tables the donor carries no class columns and the share is absent.
    plain = clean_nts_travel_tables(household, individual, trip, columns=_columns())
    assert plain.receipt["ticket_tables_read"] is False
    assert boarding_class_column(BUS_IN_LONDON, BOARDING_FARE_PAID) not in plain.person
    bad = dict(_operation(_committed_stage(), "clean_nts_travel_tables"))
    bad["codes"] = {**bad["codes"], "concessionary_ticket": []}
    with pytest.raises(NTSBusTravelError, match="concessionary_ticket"):
        NTSColumns.from_parameters(bad)


def test_ticket_receipt_weights_boardings_by_the_full_trip_weight() -> None:
    """W5 carries the household weight W2: the receipt must not divide it out."""

    household, individual, trip, stage, ticket = _synthetic_nts_with_tickets()
    # Season-ticket households stand for five times as many households.
    season = set(ticket.loc[ticket["SpecialTicket_B01ID"] == 1, "IndividualID"])
    heavy = set(individual.loc[individual["IndividualID"].isin(season), "HouseholdID"])
    household.loc[household["HouseholdID"].isin(heavy), "W2"] *= 5.0
    trip.loc[trip["HouseholdID"].isin(heavy), "W5"] *= 5.0
    donor = clean_nts_travel_tables(
        household, individual, trip, stage, ticket, columns=_columns()
    )
    by_series = donor.receipt["boarding_classes"]["by_series"]
    trip_weight = stage["TripID"].map(trip.set_index("TripID")["W5"])
    weight = trip_weight * stage["NumBoardings"]
    special = stage["IndTicketID"].map(
        ticket.set_index("IndTicketID")["SpecialTicket_B01ID"]
    )
    paid = stage["IndTicketID"].isna() & (stage["StageFareCost"] > 0)
    passes = special.notna() & ~special.isin([7, 8, 9, 10, 11])
    for series, mode in ((BUS_IN_LONDON, 7), (OTHER_LOCAL_BUS, 8)):
        cell = stage["StageMode_B04ID"] == mode
        single = weight[cell & paid].sum()
        assert by_series[series][
            "single_share_of_fare_paying_boardings"
        ] == pytest.approx(single / (single + weight[cell & passes].sum()))
        # One boarding stage per synthetic trip.
        assert by_series[series]["boardings_per_bus_trip"] == pytest.approx(
            weight[cell].sum() / trip_weight[cell].sum()
        )


def _single_fare_donor() -> pd.DataFrame:
    """One band-3 person per survey year and group, with chosen boardings."""

    rows = []
    for year, group, paid, passes in (
        (2022, REST_OF_ENGLAND_GROUP, 0.0, 10.0),
        (2023, REST_OF_ENGLAND_GROUP, 8.0, 2.0),
        (2024, REST_OF_ENGLAND_GROUP, 6.0, 4.0),
        (2022, LONDON_GROUP, 1.0, 9.0),
        (2023, LONDON_GROUP, 5.0, 5.0),
        (2024, LONDON_GROUP, 3.0, 7.0),
    ):
        own = BUS_IN_LONDON if group == LONDON_GROUP else OTHER_LOCAL_BUS
        row = {
            "survey_year": year,
            "residence_group": group,
            "weight": 1.0,
            "local_bus_use_band": 3,
        }
        for series in (BUS_IN_LONDON, OTHER_LOCAL_BUS):
            row[boarding_class_column(series, BOARDING_FARE_PAID)] = (
                paid if series == own else 0.0
            )
            row[boarding_class_column(series, BOARDING_PASS)] = (
                passes if series == own else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_single_fare_share_reads_the_cap_years_only() -> None:
    """2022, before the cap, would pull the pooled share down (microcosm#1331)."""

    donor = _single_fare_donor()
    pooled, pooled_receipt = single_fare_shares_from_donor(donor)
    capped, capped_receipt = single_fare_shares_from_donor(donor, years=(2023, 2024))
    assert pooled[OTHER_LOCAL_BUS][REST_OF_ENGLAND_GROUP][3] == pytest.approx(14 / 30)
    assert capped[OTHER_LOCAL_BUS][REST_OF_ENGLAND_GROUP][3] == pytest.approx(14 / 20)
    assert pooled[BUS_IN_LONDON][LONDON_GROUP][3] == pytest.approx(9 / 30)
    assert capped[BUS_IN_LONDON][LONDON_GROUP][3] == pytest.approx(8 / 20)
    assert pooled_receipt["survey_years"] == [2022, 2023, 2024]
    assert capped_receipt["survey_years"] == [2023, 2024]
    with pytest.raises(NTSBusTravelError, match="survey years"):
        single_fare_shares_from_donor(donor, years=(2021,))


def _fare_mix(
    stage: pd.DataFrame,
    household: pd.DataFrame,
    individual: pd.DataFrame,
    trip: pd.DataFrame,
    ticket: pd.DataFrame,
) -> dict:
    regime = FareCapRegime.from_parameters(
        _operation(_committed_stage(), "clean_nts_travel_tables"),
        survey_years=_columns().survey_years,
    )
    donor = clean_nts_travel_tables(
        household, individual, trip, stage, ticket, columns=_columns(), fare_cap=regime
    )
    return donor.receipt["boarding_classes"]["fare_mix"]


def test_fare_mix_lines_report_aggregates_and_withhold_thin_cells() -> None:
    """Before the cap: share above it and their mean fare; under it: share at it."""

    household, individual, trip, stage, ticket = _synthetic_nts_with_tickets(180)
    year = stage["HouseholdID"].map(household.set_index("HouseholdID")["SurveyYear"])
    region = stage["HouseholdID"].map(
        household.set_index("HouseholdID")["HHoldGOR_B02ID"]
    )
    rest = (region != 7).to_numpy()
    point_of_use = (
        stage["IndTicketID"].isna() & (stage["StageFareCost"] > 0)
    ).to_numpy()
    # Before the cap, the full single fares sit above £2.
    raised = stage.copy()
    full = (
        rest
        & point_of_use
        & (year == 2022).to_numpy()
        & (stage["StageFareCost"] == 2.0).to_numpy()
    )
    raised.loc[full, "StageFareCost"] = 2.75
    lines = _fare_mix(raised, household, individual, trip, ticket)
    # The expected values, on the full trip weight (W5 x JJXSC, which carries
    # the household weight W2) times the boardings, and the fare per boarding.
    weight = (
        raised["TripID"].map(trip.set_index("TripID")["W5"]) * raised["NumBoardings"]
    ).to_numpy()
    fare = (raised["StageFareCost"] / raised["NumBoardings"]).to_numpy()
    base = rest & point_of_use & (year == 2022).to_numpy()
    above = base & (fare > 2.01)
    assert lines["before_cap"]["2022"]["share_above_cap"] == round(
        weight[above].sum() / weight[base].sum(), 2
    )
    assert lines["before_cap"]["2022"]["mean_fare_above_cap"] == round(
        float(np.average(fare[above], weights=weight[above])), 2
    )
    for y in (2023, 2024):
        base = rest & point_of_use & (year == y).to_numpy()
        at = base & (np.abs(fare - 2.0) <= 0.01)
        assert lines["under_cap"][str(y)]["share_at_cap"] == round(
            weight[at].sum() / weight[base].sum(), 2
        )
    assert set(lines["before_cap"]["2022"]) == {
        "share_above_cap",
        "mean_fare_above_cap",
    }
    assert set(lines["under_cap"]["2023"]) == {"share_at_cap"}
    assert lines["minimum_cell"] == {"boardings": 3, "households": 3}
    # Above the cap on two households only: both 2022 lines are withheld,
    # however many boardings those households make.
    thin = stage.copy()
    few = (
        stage["HouseholdID"]
        .isin(sorted(set(stage.loc[full, "HouseholdID"]))[:2])
        .to_numpy()
    )
    thin.loc[full & few, "StageFareCost"] = 2.75
    lines = _fare_mix(thin, household, individual, trip, ticket)
    assert lines["before_cap"]["2022"] == {
        "share_above_cap": UK_NTS_RECEIPT_SUPPRESSED,
        "mean_fare_above_cap": UK_NTS_RECEIPT_SUPPRESSED,
    }
    # Under the cap, a share whose complement rests on two households is
    # withheld too: one minus it would give that thin cell back.
    capped = stage.copy()
    in_2023 = rest & point_of_use & (year == 2023).to_numpy()
    capped.loc[in_2023, "StageFareCost"] = 2.0 * capped.loc[in_2023, "NumBoardings"]
    off_cap = sorted(set(stage.loc[in_2023, "HouseholdID"]))[:2]
    capped.loc[
        in_2023 & stage["HouseholdID"].isin(off_cap).to_numpy(), "StageFareCost"
    ] = 1.5
    lines = _fare_mix(capped, household, individual, trip, ticket)
    assert lines["under_cap"]["2023"]["share_at_cap"] == UK_NTS_RECEIPT_SUPPRESSED
    assert lines["under_cap"]["2024"]["share_at_cap"] != UK_NTS_RECEIPT_SUPPRESSED


def test_cleaning_handles_the_nts_sentinels_and_the_diary_sample() -> None:
    """-9 is a declared non-user, -8 and unknown age codes drop, W2 = 0 leaves."""

    household, individual, trip = _synthetic_nts()
    individual = individual.copy()
    first, second, third = individual.index[:3]
    individual.loc[first, "OrdBus2Freq_B01ID"] = -9
    individual.loc[second, "OrdBus2Freq_B01ID"] = -8
    individual.loc[third, "Age_B01ID"] = -8
    household = household.copy()
    # An interview-only household: W2 of zero, so it leaves the donor.
    zero = household.index[-1]
    household.loc[zero, "W2"] = 0.0
    zero_id = int(household.loc[zero, "HouseholdID"])
    donor = clean_nts_travel_tables(household, individual, trip, columns=_columns())
    person = donor.person
    ids = set(person["individual_id"])
    assert int(individual.loc[first, "IndividualID"]) in ids
    assert (
        person.loc[
            person["individual_id"] == int(individual.loc[first, "IndividualID"]),
            "local_bus_use_band",
        ].item()
        == 0
    )
    assert int(individual.loc[second, "IndividualID"]) not in ids
    assert int(individual.loc[third, "IndividualID"]) not in ids
    assert zero_id not in set(person["household_id"])
    assert donor.receipt["persons_unmapped_frequency"] == 1
    assert donor.receipt["persons_unmapped_age"] == 1
    assert donor.receipt["households_zero_weight"] == 1
    bad = dict(_operation(_committed_stage(), "clean_nts_travel_tables"))
    bad["trip_weight_basis"] = "per_stage"
    with pytest.raises(NTSBusTravelError, match="trip_weight_basis"):
        NTSColumns.from_parameters(bad)
    bad = dict(_operation(_committed_stage(), "clean_nts_travel_tables"))
    bad["codes"] = {**bad["codes"], "age_band": {"1": 5}}
    with pytest.raises(NTSBusTravelError, match="codes.age_band"):
        NTSColumns.from_parameters(bad)


def test_cleaning_falls_back_to_published_shares_without_a_frequency_column() -> None:
    household, individual, trip = _synthetic_nts(frequency=False)
    donor = clean_nts_travel_tables(household, individual, trip, columns=_columns())
    assert donor.frequency_source == "published_shares"
    assert "local_bus_use_band" not in donor.person
    assert donor.receipt["frequency_column"] == "ordbus2freq_b01id"


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
        # The band reads HBAI net income plus the HBAI deductions (#1164).
        rows = len(frame.table("household"))
        values = {
            "hbai_household_net_income_before_reset_to_zero": self.rng.uniform(
                5e3, 9e4, rows
            )
        }
        return {name: values.get(name, np.zeros(rows)) for name in variables}


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


def test_recipient_income_band_is_hbai_gross_income() -> None:
    # microcosm#1164: the NTS band is gross cash income, so the recipient adds
    # the engine's HBAI deductions back to its HBAI net income; the engine's
    # household_gross_income (capital gains, imputed NHS and education) is not
    # asked for.
    from microcosm.build.uk_runtime.household_income import (
        UK_HBAI_GROSS_INCOME_DEDUCTIONS,
    )

    frame, _ = _recipient_frame()
    rows = len(frame.table("household"))
    net = np.full(rows, 20_000.0)
    net[: rows // 2] = 40_000.0
    requested: list[str] = []

    class _GrossEngine(_Engine):
        def materialize(self, frame, variables, period):
            requested.extend(variables)
            values = {
                "hbai_household_net_income_before_reset_to_zero": net,
                "income_tax": np.full(rows, 6_000.0),
            }
            return {name: values.get(name, np.zeros(rows)) for name in variables}

    band_parameters = _operation(_committed_stage(), IMPUTE_BUS_USE_BAND_KIND)
    recipient = recipient_predictors(
        _recipient_frame()[0],
        _GrossEngine(np.random.default_rng(0)),
        columns=_columns(),
        income_band_edges=band_parameters["income_band_edges"],
        region_remap=band_parameters.get("region_remap", {}),
    )

    assert "household_gross_income" not in requested
    assert set(UK_HBAI_GROSS_INCOME_DEDUCTIONS) <= set(requested)
    # 20,000 + 6,000 and 40,000 + 6,000 both fall in band 2 (25,000 to
    # 50,000); on the net income alone the first would be band 1.
    assert set(recipient["income_band"]) == {2}


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
    # microcosm#1386, option 1: no group's non-user band carries trips, and
    # the other bands keep each group's diary trips per person on the donor.
    person = donor.person
    weights = person["weight"].to_numpy(dtype=float)
    groups_of = person["residence_group"].to_numpy().astype(str)
    for series, column in SERIES_TRIP_COLUMNS.items():
        for group in (LONDON_GROUP, REST_OF_ENGLAND_GROUP):
            assert means[series][group][NON_USER_BAND] == 0.0
            in_group = groups_of == group
            diary = np.average(
                person[column].to_numpy(dtype=float)[in_group],
                weights=weights[in_group],
            )
            assigned = np.average(
                [
                    means[series][group][int(b)]
                    for b in person["local_bus_use_band"].to_numpy()[in_group]
                ],
                weights=weights[in_group],
            )
            assert assigned == pytest.approx(diary)
    assert receipt["non_user_band"]["rule"] == NON_USER_BAND_RULE
    assert set(receipt["non_user_band"]["other_band_factors"]) == set(means)
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
    # Under-5s ride free everywhere; Northern Ireland's SmartPass is 60+ (the
    # 65+ pass adds all-Ireland travel only); England outside London is State
    # Pension age.
    ages = np.array([4.0, 10.0, 17.0, 18.0, 21.0, 22.0, 59.0, 60.0, 65.0, 66.0])
    for region, expected in (
        (
            "SOUTH_EAST",
            [True, False, False, False, False, False, False, False, False, True],
        ),
        ("LONDON", [True, True, True, False, False, False, False, True, True, True]),
        ("SCOTLAND", [True, True, True, True, True, False, False, True, True, True]),
        ("WALES", [True, False, False, False, False, False, False, True, True, True]),
        (
            "NORTHERN_IRELAND",
            [True, False, False, False, False, False, False, True, True, True],
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
    household, individual, trip, nts_stage, nts_ticket = _synthetic_nts_with_tickets()
    frame, rng = _recipient_frame()
    transform = UKNTSBusTravelStageTransform(
        stage=stage,
        engine=_Engine(rng),
        nts_household=household,
        nts_individual=individual,
        nts_trip=trip,
        nts_stage=nts_stage,
        nts_ticket=nts_ticket,
    )
    result = transform(frame)
    person = result.table("person")
    out_household = result.table("household")
    assert set(UK_NTS_BUS_TRAVEL_OUTPUT_COLUMNS) <= set(person.columns) | set(
        out_household.columns
    )
    assert person["local_bus_use_band"].between(NON_USER_BAND, MAX_BAND).all()
    assert person["local_bus_single_fare_share"].between(0.0, 1.0).all()
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
    # Outside London and Scotland nobody aged 5 to 59 is eligible (England's
    # rule is State Pension age, Wales and Northern Ireland 60); under-5s are
    # everywhere, and Scotland's under-22s are.
    plain = region.isin(["SOUTH_EAST", "NORTH_WEST", "WALES", "NORTHERN_IRELAND"])
    grown = plain & (person["age"] >= 5) & (person["age"] < 60)
    assert (~person.loc[grown, "bus_pass_eligible"]).all()
    assert person.loc[person["age"] < 5, "bus_pass_eligible"].all()
    assert person.loc[
        region.isin(["WALES", "NORTHERN_IRELAND"]) & (person["age"] >= 60),
        "bus_pass_eligible",
    ].all()
    # The stage's outputs sit at the end of the person table in the declared
    # order (the graph's cell order; the H2 parity identity depends on it).
    tail = list(person.columns)[-len(UK_NTS_PERSON_OUTPUT_COLUMNS) :]
    assert tail == list(UK_NTS_PERSON_OUTPUT_COLUMNS)
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
    shares = evidence["band_means"]["single_fare_shares"]
    assert shares["survey_years"] == [2023, 2024]
    fare_mix = evidence["donor"]["boarding_classes"]["fare_mix"]
    assert set(fare_mix["before_cap"]) == {"2022"}
    assert set(fare_mix["under_cap"]) == {"2023", "2024"}
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
