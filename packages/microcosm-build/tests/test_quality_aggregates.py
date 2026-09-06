"""Public population aggregates use the actual Frame's typed weight resolution."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.quality_aggregates import (
    DistributionBin,
    DistributionSpec,
    GeographySpec,
    population_distributions,
    population_support,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights, wsum

IDENTITY = {
    "candidate_release_id": "candidate-2024-test",
    "population_sha256": "a" * 64,
    "period": "2024",
}
GEOGRAPHIES = (
    GeographySpec("national", None, "2024", {"us": "United States"}),
    GeographySpec(
        "state",
        "state_fips",
        "2024",
        {"6": "California", "36": "New York", "48": "Texas", "unknown": "Unassigned"},
        unknown_id="unknown",
    ),
    GeographySpec(
        "congressional_district",
        "cd",
        "119",
        {
            "601": "CA-01",
            "602": "CA-02",
            "3601": "NY-01",
            "4801": "TX-01",
            "unknown": "Unassigned",
        },
        unknown_id="unknown",
    ),
)
INCOME = DistributionSpec(
    variable="income",
    label="Personal income",
    entity="person",
    unit="USD/year (2024)",
    bin_definition_id="income-2024-test-v1",
    bins=(
        DistributionBin("Below 10", None, 10),
        DistributionBin("10–20", 10, 20),
        DistributionBin("20–40", 20, 40),
    ),
)


@pytest.fixture
def frame():
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2, 3, 4, 5, 6, 7],
                    "person_household_id": [10, 10, 20, 20, 20, 30, 40],
                    "income": [0.0, 10.0, 20.0, 30.0, np.nan, 15.0, 40.0],
                    "ones": [1.0] * 7,
                },
                index=[70, 20, 11, 12, 15, 19, 5],
            ),
            "household": pd.DataFrame(
                {
                    "household_id": [10, 20, 30, 40],
                    "state_fips": pd.array([6, 36, 6, None], dtype="Int64"),
                    "cd": pd.array([601, 3601, 602, None], dtype="Int64"),
                    "wealth": [100.0, 200.0, 150.0, np.nan],
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([2.0, 6.0, 0.0, 4.0]), WeightKind.CALIBRATED)},
    )


def support(frame, **kwargs):
    return population_support(
        frame, **IDENTITY, entity="person", geography_specs=GEOGRAPHIES, **kwargs
    )


def distributions(frame, *, specs=(INCOME,), geographies=GEOGRAPHIES):
    return population_distributions(
        frame, **IDENTITY, distributions=specs, geography_specs=geographies
    )


def row(document, level="national", identifier="us"):
    return next(
        row
        for row in document.get("rows", document.get("series"))
        if row["geography"]["level"] == level and row["geography"]["id"] == identifier
    )


def rebuild(frame, tables, weights=None):
    return Frame(
        tables,
        frame.schema,
        weights
        or {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
    )


def test_support_matches_hand_calculation_and_frame_reducer(frame):
    result = support(frame)
    assert result["schema_version"] == "microcosm.population_support.v1"
    assert {key: result[key] for key in IDENTITY} == IDENTITY
    assert result["weight_entity"] == "household"
    assert result["weight_basis"] == "calibrated:household->person"
    national = row(result)
    assert national["record_count"] == 7
    assert national["positive_weight_count"] == 6
    assert national["weight_sum"] == wsum(frame, "ones") == 26
    assert national["effective_sample_size"] == pytest.approx(169 / 33)
    assert national["max_weight_share"] == pytest.approx(3 / 13)
    for level in ("state", "congressional_district"):
        partition = [
            item for item in result["rows"] if item["geography"]["level"] == level
        ]
        assert sum(item["weight_sum"] for item in partition) == national["weight_sum"]
        assert (
            sum(item["record_count"] for item in partition) == national["record_count"]
        )
    assert row(result, "state", "unknown")["weight_sum"] == 4
    for level, identifier, count in [
        ("state", "48", 0),
        ("congressional_district", "602", 1),
    ]:
        empty = row(result, level, identifier)
        assert empty["record_count"] == count
        assert empty["weight_sum"] == empty["positive_weight_count"] == 0
        assert empty["effective_sample_size"] is empty["max_weight_share"] is None


def test_household_support_does_not_count_each_person(frame):
    result = population_support(
        frame, **IDENTITY, entity="household", geography_specs=GEOGRAPHIES
    )
    assert result["weight_basis"] == "calibrated:household"
    assert row(result)["record_count"] == 4
    assert row(result)["weight_sum"] == 12
    assert row(result)["effective_sample_size"] == pytest.approx(18 / 7)


def test_distributions_include_missing_mass_in_denominator_and_final_boundary(frame):
    result = distributions(frame)
    national = row(result)
    assert national["total_weight"] == 26
    assert national["missing_weight"] == 6
    assert [item["weight"] for item in national["bins"]] == [2, 2, 16]
    assert [item["share"] for item in national["bins"]] == pytest.approx(
        [2 / 26, 2 / 26, 16 / 26]
    )
    assert sum(item["share"] for item in national["bins"]) == pytest.approx(20 / 26)
    assert row(result, "state", "36")["missing_weight"] == 6
    for level in ("state", "congressional_district"):
        partition = [
            item for item in result["series"] if item["geography"]["level"] == level
        ]
        assert sum(item["total_weight"] for item in partition) == 26
        for index in range(3):
            assert (
                sum(item["bins"][index]["weight"] for item in partition)
                == national["bins"][index]["weight"]
            )
    zero = row(result, "congressional_district", "602")
    assert zero["total_weight"] == zero["missing_weight"] == 0
    assert all(item["share"] is None for item in zero["bins"])


def test_distributions_on_different_entities_keep_their_weight_basis(frame):
    wealth = DistributionSpec(
        "wealth",
        "Household wealth",
        "household",
        "USD (2024)",
        "wealth-v1",
        (DistributionBin("Below 150", None, 150), DistributionBin("150+", 150, None)),
    )
    result = distributions(frame, specs=(INCOME, wealth))
    household = next(
        item
        for item in result["series"]
        if item["variable"] == "wealth" and item["geography"]["level"] == "national"
    )
    assert household["total_weight"] == 12
    assert household["missing_weight"] == 4
    assert household["weight_basis"] == "calibrated:household"
    assert [item["weight"] for item in household["bins"]] == [2, 6]


def test_row_permutation_preserves_every_aggregate(frame):
    # Frame requires group rows in sorted id order; person order is free.
    permuted = rebuild(
        frame,
        {
            "person": frame.person.iloc[::-1],
            "household": frame.table("household"),
        },
    )
    assert support(permuted) == support(frame)
    assert distributions(permuted) == distributions(frame)


def test_explicit_person_weights_are_resolved_by_frame(frame):
    explicit = rebuild(
        frame,
        {entity: frame.table(entity) for entity in frame.entities},
        {
            "household": frame.weights_for("household"),
            "person": Weights(np.ones(7), WeightKind.DESIGN),
        },
    )
    result = support(explicit)
    assert result["weight_entity"] == "person"
    assert result["weight_basis"] == "design:person"
    assert row(result)["weight_sum"] == 7


@pytest.mark.parametrize("function", [support, distributions])
def test_undeclared_missing_geography_refuses_instead_of_dropping_mass(frame, function):
    frame.table("household").loc[0, "state_fips"] = 99
    with pytest.raises(ValueError, match="geography.*99"):
        function(frame)


def test_missing_geography_requires_an_explicit_unknown_bucket(frame):
    no_unknown = replace(GEOGRAPHIES[1], unknown_id=None)
    with pytest.raises(ValueError, match="missing.*geography"):
        distributions(frame, geographies=(GEOGRAPHIES[0], no_unknown))


def test_vintages_remain_distinct_even_for_the_same_geographic_ids(frame):
    # Separate producer-owned columns carry each vintage; equal GEOIDs do not
    # imply comparable boundaries or authorize reusing one vintage's values.
    frame.table("household")["cd_117"] = pd.array([602, 3601, 601, None], dtype="Int64")
    old = replace(GEOGRAPHIES[2], column="cd_117", vintage="117")
    result = distributions(frame, geographies=(*GEOGRAPHIES, old))
    vintages = {
        item["geography"]["vintage"]
        for item in result["series"]
        if item["geography"]["level"] == "congressional_district"
    }
    assert vintages == {"117", "119"}
    masses = {
        item["geography"]["vintage"]: item["total_weight"]
        for item in result["series"]
        if item["geography"]["level"] == "congressional_district"
        and item["geography"]["id"] == "601"
    }
    assert masses == {"117": 0, "119": 4}


@pytest.mark.parametrize(
    "bins",
    [
        (DistributionBin("Incomplete", 1, None),),
        (DistributionBin("Incomplete", None, 39),),
    ],
)
def test_non_exhaustive_bins_refuse_even_if_only_zero_weight_value_is_omitted(
    frame, bins
):
    with pytest.raises(ValueError, match="cover"):
        distributions(frame, specs=(replace(INCOME, bins=bins),))
    frame.person.loc[frame.person["income"] == 15, "income"] = 1000
    with pytest.raises(ValueError, match="cover"):
        distributions(frame)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_values_refuse(frame, value):
    frame.person.iloc[0, frame.person.columns.get_loc("income")] = value
    with pytest.raises(ValueError, match="infinite"):
        distributions(frame)


def test_cannot_choose_an_arbitrary_persons_geography_for_a_household(frame):
    moved = frame.place("state_fips", "person", how="broadcast")
    with pytest.raises(ValueError, match="align.*geography"):
        population_support(
            moved, **IDENTITY, entity="household", geography_specs=GEOGRAPHIES
        )


def test_invalid_identity_and_declaration_fields_refuse(frame):
    with pytest.raises(ValueError, match="digest"):
        population_support(
            frame,
            **(IDENTITY | {"population_sha256": "node-id"}),
            entity="person",
            geography_specs=GEOGRAPHIES,
        )
    with pytest.raises(ValueError, match="vintage"):
        replace(GEOGRAPHIES[2], vintage="")
    with pytest.raises(ValueError, match="national"):
        distributions(frame, geographies=GEOGRAPHIES[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        distributions(frame, specs=(INCOME, INCOME))
    with pytest.raises(ValueError, match="contiguous"):
        replace(
            INCOME,
            bins=(DistributionBin("A", None, 10), DistributionBin("B", 11, None)),
        )
    with pytest.raises(ValueError, match="entity"):
        distributions(frame, specs=(replace(INCOME, entity="household"),))


def test_all_missing_values_are_visible_and_inputs_are_unchanged(frame):
    frame.person["income"] = pd.array([None] * 7, dtype="Float64")
    before = frame.person.copy(deep=True)
    national = row(distributions(frame))
    assert national["missing_weight"] == national["total_weight"] == 26
    assert all(item["weight"] == item["share"] == 0 for item in national["bins"])
    pd.testing.assert_frame_equal(before, frame.person)


def test_ess_is_stable_when_squaring_raw_weights_would_overflow(frame):
    scaled = rebuild(
        frame,
        {entity: frame.table(entity) for entity in frame.entities},
        {
            "household": Weights(
                frame.weights_for("household").values * 1e200, WeightKind.CALIBRATED
            )
        },
    )
    original, large = row(support(frame)), row(support(scaled))
    assert large["effective_sample_size"] == pytest.approx(
        original["effective_sample_size"]
    )
    assert large["max_weight_share"] == pytest.approx(original["max_weight_share"])
    assert large["weight_sum"] == pytest.approx(original["weight_sum"] * 1e200)


@pytest.mark.parametrize("values", [["1"] * 7, [1 + 2j] * 7])
def test_unsupported_variable_types_cannot_be_silently_coerced(frame, values):
    frame.person["income"] = values
    with pytest.raises(ValueError, match="real numeric"):
        distributions(frame)


def test_stale_reserved_weight_column_cannot_override_typed_weights(frame):
    frame.person["person_weight"] = 999.0
    with pytest.raises(ValueError, match="person_weight"):
        support(frame)


def test_same_level_and_vintage_cannot_appear_twice(frame):
    with pytest.raises(ValueError, match="Duplicate geography"):
        distributions(frame, geographies=(*GEOGRAPHIES, GEOGRAPHIES[1]))


def test_geography_ids_are_explicitly_normalized_without_losing_string_padding(frame):
    frame.table("household")["state_fips"] = ["06", "36", "06", None]
    geo = replace(
        GEOGRAPHIES[1],
        labels={"06": "California", "36": "New York", "unknown": "Unassigned"},
    )
    result = distributions(frame, geographies=(GEOGRAPHIES[0], geo))
    assert row(result, "state", "06")["total_weight"] == 4


def test_boolean_distribution_and_nullable_missingness(frame):
    frame.person["income"] = pd.array(
        [False, True, True, False, None, False, True], dtype="boolean"
    )
    spec = replace(
        INCOME,
        unit="indicator",
        bins=(DistributionBin("False", 0, 0.5), DistributionBin("True", 0.5, 1)),
    )
    result = row(distributions(frame, specs=(spec,)))
    assert [item["weight"] for item in result["bins"]] == [8, 12]
    assert result["missing_weight"] == 6


@pytest.mark.parametrize(
    "level",
    [
        "region",
        "constituency",
        "local_authority",
        "nuts1",
        "nuts2",
        "commune",
        "producer_area_7",
    ],
)
def test_producer_declares_canonical_levels_without_country_inference(level):
    spec = GeographySpec(level, "area", "fixture-v1", {"001": "Invented area"})
    assert spec.level == level


@pytest.mark.parametrize(
    "level",
    [
        None,
        1,
        True,
        "",
        "Region",
        " region",
        "region ",
        "local-authority",
        "local authority",
        "nuts.1",
        "_region",
        "region_",
        "region__one",
        "1region",
        "région",
    ],
)
def test_noncanonical_level_identifiers_refuse(level):
    with pytest.raises(ValueError, match="level"):
        GeographySpec(level, "area", "fixture-v1", {"001": "Invented area"})


@pytest.mark.parametrize(
    "change",
    [
        {"labels": {}},
        {"labels": {"": "Area"}},
        {"labels": {"1": ""}},
        {"labels": {1: "Area"}},
        {"vintage": None},
        {"column": None},
        {"unknown_id": "absent"},
        {"unknown_id": True},
    ],
)
def test_area_definitions_require_complete_explicit_metadata(change):
    with pytest.raises(ValueError):
        replace(GEOGRAPHIES[1], **change)


@pytest.mark.parametrize(
    "change",
    [
        {"column": "region"},
        {"labels": {"uk": "UK", "be": "BE"}},
        {"unknown_id": "us"},
    ],
)
def test_national_special_rules_remain_strict(change):
    with pytest.raises(ValueError, match="national"):
        replace(GEOGRAPHIES[0], **change)


@pytest.fixture(params=["uk", "be"])
def invented_country(request):
    # Invented engineering rows and labels; these are not country source data,
    # an authoritative boundary catalog, or evidence of analysis suitability.
    country = request.param
    levels = (
        ("region", "constituency", "local_authority")
        if country == "uk"
        else ("nuts1", "nuts2", "commune")
    )
    data = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2, 3, 4, 5, 6],
                    "person_household_id": [10, 10, 10, 20, 20, 30],
                    "person_benunit_id": [1, 1, 2, 3, 3, 4],
                    "income": [-1.0, 0.0, 10.0, np.nan, 20.0, 0.0],
                },
                index=[50, 10, 30, 70, 90, 20],
            ),
            "household": pd.DataFrame(
                {
                    "household_id": [10, 20, 30],
                    "broad_area": ["001", "002", None],
                    "local_area": ["01001", "02002", None],
                    "old_local_area": ["02002", "01001", None],
                    "fine_area": ["F1", "F2", "unassigned"],
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": [1, 2, 3, 4],
                    "benunit_area": ["001", "001", "002", None],
                    "resources": [0.0, 10.0, np.nan, 20.0],
                }
            ),
        },
        EntitySchema(group_entities=("household", "benunit")),
        {"household": Weights(np.array([2.0, 5.0, 3.0]), WeightKind.DESIGN)},
    )
    geos = (
        GeographySpec(
            "national",
            None,
            "fixture-2024",
            {country: f"Invented {country.upper()} fixture"},
        ),
        GeographySpec(
            levels[0],
            "broad_area",
            "fixture-2024",
            {
                "001": "Area A",
                "002": "Area B",
                "003": "Empty area",
                "unknown": "Unknown area",
            },
            "unknown",
        ),
        GeographySpec(
            levels[1],
            "local_area",
            "fixture-2024",
            {"01001": "Local A", "02002": "Local B", "unknown": "Unknown local area"},
            "unknown",
        ),
        GeographySpec(
            levels[1],
            "old_local_area",
            "fixture-2021",
            {
                "01001": "Old local A",
                "02002": "Old local B",
                "unknown": "Unknown old area",
            },
            "unknown",
        ),
        GeographySpec(
            levels[2],
            "fine_area",
            "fixture-2025",
            {"F1": "Fine A", "F2": "Fine B", "unassigned": "Explicitly unassigned"},
        ),
    )
    return data, geos, country, levels


def test_country_partitions_conserve_support_and_every_distribution_bin(
    invented_country,
):
    frame, geos, country, levels = invented_country
    result = population_support(
        frame, **IDENTITY, entity="person", geography_specs=geos
    )
    dist = distributions(frame, geographies=geos)
    national = row(result, identifier=country)
    assert national["record_count"] == 6
    assert national["weight_sum"] == 19  # 3*2 + 2*5 + 1*3 inherited household weights.
    assert result["weight_basis"] == "design:household->person"
    assert row(result, levels[0], "unknown")["weight_sum"] == 3
    assert row(result, levels[0], "003")["record_count"] == 0
    assert row(result, levels[2], "unassigned")["weight_sum"] == 3
    for spec in geos:
        partition = [
            item
            for item in result["rows"]
            if (item["geography"]["level"], item["geography"]["vintage"])
            == (spec.level, spec.vintage)
        ]
        series = [
            item
            for item in dist["series"]
            if (item["geography"]["level"], item["geography"]["vintage"])
            == (spec.level, spec.vintage)
        ]
        assert sum(item["record_count"] for item in partition) == 6
        assert sum(item["weight_sum"] for item in partition) == 19
        assert sum(item["missing_weight"] for item in series) == 5
        assert [
            sum(item["bins"][i]["weight"] for item in series) for i in range(3)
        ] == [7, 2, 5]
    local = {
        item["geography"]["vintage"]: item["weight_sum"]
        for item in result["rows"]
        if item["geography"]["level"] == levels[1]
        and item["geography"]["id"] == "01001"
    }
    assert local == {"fixture-2024": 6, "fixture-2021": 10}
    assert "country" not in result and "release_eligible" not in result


def test_group_weights_are_inherited_once_per_group_without_geography_shortcuts(
    invented_country,
):
    frame, geos, country, levels = invented_country
    group_geos = (geos[0], replace(geos[1], column="benunit_area"))
    result = population_support(
        frame, **IDENTITY, entity="benunit", geography_specs=group_geos
    )
    assert result["weight_entity"] == "household"
    assert result["weight_basis"] == "design:household->person->benunit"
    assert (
        row(result, identifier=country)["weight_sum"] == 12
    )  # 2+2+5+3, once per benunit.
    assert row(result, identifier=country)["record_count"] == 4
    spec = replace(INCOME, variable="resources", entity="benunit", unit="fixture units")
    dist = row(
        distributions(frame, specs=(spec,), geographies=group_geos), identifier=country
    )
    assert dist["total_weight"] == 12 and dist["missing_weight"] == 5
    assert [item["weight"] for item in dist["bins"]] == [2, 2, 3]
    with pytest.raises(ValueError, match="align.*geography"):
        population_support(frame, **IDENTITY, entity="benunit", geography_specs=geos)


def test_country_missing_undeclared_and_duplicate_geographies_refuse(invented_country):
    frame, geos, _, _ = invented_country
    with pytest.raises(ValueError, match="missing.*geography"):
        distributions(frame, geographies=(geos[0], replace(geos[1], unknown_id=None)))
    with pytest.raises(ValueError, match="Undeclared"):
        distributions(
            frame,
            geographies=(
                geos[0],
                replace(
                    geos[4],
                    labels={"F1": "A", "F2": "B", "unknown": "Unknown"},
                    unknown_id="unknown",
                ),
            ),
        )
    with pytest.raises(ValueError, match="Duplicate geography"):
        distributions(frame, geographies=(*geos, replace(geos[1], column="local_area")))
    with pytest.raises(ValueError, match="not found"):
        distributions(frame, geographies=(geos[0], replace(geos[1], column="absent")))
    with pytest.raises(ValueError, match="Exactly one"):
        distributions(
            frame, geographies=(*geos, replace(geos[0], vintage="fixture-2021"))
        )


def test_categorical_geography_accepts_explicit_unknown_without_mutating_frame(frame):
    household = frame.table("household")
    household["state_fips"] = pd.Categorical(household["state_fips"])
    before = household.copy(deep=True)
    result = population_support(
        frame, **IDENTITY, entity="household", geography_specs=GEOGRAPHIES
    )
    assert row(result, "state", "unknown")["weight_sum"] == 4
    wealth = replace(
        INCOME,
        variable="wealth",
        entity="household",
        bins=(DistributionBin("All", None, None),),
    )
    assert (
        row(distributions(frame, specs=(wealth,)), "state", "unknown")["missing_weight"]
        == 4
    )
    pd.testing.assert_frame_equal(before, household)


def test_broadcast_uses_the_schema_person_entity(frame):
    person = frame.person.rename(
        columns={
            "person_id": "individual_id",
            "person_household_id": "individual_household_id",
        }
    )
    renamed = Frame(
        {"individual": person, "household": frame.table("household")},
        EntitySchema(person_entity="individual", group_entities=("household",)),
        {"household": frame.weights_for("household")},
    )
    result = population_support(
        renamed, **IDENTITY, entity="individual", geography_specs=GEOGRAPHIES
    )
    assert row(result)["weight_sum"] == 26
    assert result["weight_basis"] == "calibrated:household->individual"
    assert (
        row(distributions(renamed, specs=(replace(INCOME, entity="individual"),)))[
            "total_weight"
        ]
        == 26
    )


@pytest.mark.parametrize("function", [support, distributions])
def test_changed_membership_is_revalidated_before_aggregation(frame, function):
    frame.person.iloc[0, frame.person.columns.get_loc("person_household_id")] = 999
    with pytest.raises(ValueError, match="household"):
        function(frame)


def test_unequal_member_weights_cannot_create_an_inherited_group_weight(
    invented_country,
):
    frame, geos, _, _ = invented_country
    explicit = rebuild(
        frame,
        {entity: frame.table(entity) for entity in frame.entities},
        {"person": Weights(np.arange(1.0, 7.0), WeightKind.IMPORTANCE)},
    )
    with pytest.raises(ValueError, match="constant|unequal"):
        population_support(
            explicit,
            **IDENTITY,
            entity="benunit",
            geography_specs=(geos[0], replace(geos[1], column="benunit_area")),
        )


def test_geography_definitions_copy_and_freeze_the_declared_catalog():
    labels = {"001": "Invented area"}
    spec = GeographySpec("region", "area", "fixture-v1", labels)
    labels["002"] = "Late mutation"
    assert dict(spec.labels) == {"001": "Invented area"}
    with pytest.raises(TypeError):
        spec.labels["001"] = "Changed label"
