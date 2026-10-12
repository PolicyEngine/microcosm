"""Held-out population-structure reporting without a country engine."""

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.population_structure_holdout import (
    ACS_2024_B09021_65_PLUS_ADULTS,
    ACS_2024_B09021_65_PLUS_LIVES_ALONE,
    ACS_2024_B12002_65_PLUS,
    SOCIAL_SECURITY_COMPONENTS,
    SSA_SNAPSHOT_2024_12_TABLE_1,
    SSA_SNAPSHOT_2024_12_TABLE_2_BY_COMPONENT,
    SSA_SNAPSHOT_2024_12_TABLE_2_TOTAL,
    population_structure_holdout_from_h5,
    population_structure_holdout_payload,
    population_structure_holdout_unavailable,
    write_population_structure_holdout,
)


def _components(n, **positive):
    values = {name: np.zeros(n) for name in SOCIAL_SECURITY_COMPONENTS}
    for name, array in positive.items():
        values[name] = np.asarray(array, dtype=np.float64)
    return values


def _status(payload, sex, label):
    row = next(
        r for r in payload["marital_status_65_plus"]["by_sex"] if r["sex"] == sex
    )
    return next(s for s in row["statuses"] if s["status"] == label)


def test_acs_counts_reproduce_the_published_shares():
    men, women = ACS_2024_B12002_65_PLUS["male"], ACS_2024_B12002_65_PLUS["female"]
    both = men + women
    assert both.total == 61_245_283
    assert both.shares()["married"] == pytest.approx(0.5616, abs=5e-5)
    assert both.shares()["widowed"] == pytest.approx(0.2019, abs=5e-5)
    assert men.shares()["married"] == pytest.approx(0.6794, abs=5e-5)
    assert women.shares()["widowed"] == pytest.approx(0.2840, abs=5e-5)
    for counts in (men, women, both):
        assert sum(counts.shares().values()) == pytest.approx(1.0)
    assert ACS_2024_B09021_65_PLUS_LIVES_ALONE / ACS_2024_B09021_65_PLUS_ADULTS == (
        pytest.approx(0.2657, abs=5e-5)
    )


def test_ssa_snapshot_constants_add_up():
    table_1 = SSA_SNAPSHOT_2024_12_TABLE_1
    assert table_1["all"].total == 68_055_000
    assert table_1["65_plus"].total == 54_998_000
    assert table_1["under_65"].total == 13_057_000
    assert table_1["65_plus"].total + table_1["under_65"].total == table_1["all"].total
    grouped = sum(
        sum(types.values())
        for types in SSA_SNAPSHOT_2024_12_TABLE_2_BY_COMPONENT.values()
    )
    # Table 2 rounds each row to thousands; the rows sum one above the total.
    assert abs(grouped - SSA_SNAPSHOT_2024_12_TABLE_2_TOTAL) <= 1
    assert set(SSA_SNAPSHOT_2024_12_TABLE_2_BY_COMPONENT) == set(
        SOCIAL_SECURITY_COMPONENTS
    )
    with pytest.raises(TypeError):
        SSA_SNAPSHOT_2024_12_TABLE_1["new"] = table_1["all"]


def _fixture():
    # Household 1: a married couple 70/68, one with retirement benefits.
    # Household 2: a widow 80 living alone, survivor benefits.
    # Household 3: a divorced man 66 living with his daughter (40), and the
    #   daughter carries an imputed benefit split across all four components.
    # Household 4: a never-married man 30 living alone, no benefits.
    return dict(
        age=np.array([70, 68, 80, 66, 40, 30]),
        is_female=np.array([False, True, True, False, True, False]),
        household_id=np.array([1, 1, 2, 3, 3, 4]),
        marital_unit_id=np.array([10, 10, 20, 30, 31, 40]),
        marital_code=np.array([1, 1, 4, 5, 7, 7]),
        components=_components(
            6,
            social_security_retirement=[20_000, 0, 0, 0, 100, 0],
            social_security_survivors=[0, 0, 15_000, 0, 100, 0],
            social_security_dependents=[0, 0, 0, 0, 100, 0],
            social_security_disability=[0, 0, 0, 0, 100, 0],
        ),
        weights=np.array([2.0, 2.0, 3.0, 5.0, 5.0, 7.0]),
    )


def test_fixture_shares_counts_and_structure(tmp_path):
    payload = population_structure_holdout_payload(
        **_fixture(), period=2024, release_id="fixture"
    )
    # People 65+: the couple (2 + 2), the widow (3), the divorced man (5).
    assert _status(payload, "all", "married")["share"] == pytest.approx(4 / 12)
    assert _status(payload, "all", "widowed")["share"] == pytest.approx(3 / 12)
    assert _status(payload, "all", "divorced")["share"] == pytest.approx(5 / 12)
    assert _status(payload, "female", "widowed")["share"] == pytest.approx(3 / 5)
    assert _status(payload, "male", "married")["share"] == pytest.approx(2 / 7)
    married = _status(payload, "all", "married")
    assert married["difference_percentage_points"] == pytest.approx(
        100 * (4 / 12 - married["benchmark_share"])
    )
    alone = payload["living_alone_65_plus"]
    assert alone["weighted_people_65_plus"] == 12
    assert alone["share"] == pytest.approx(3 / 12)

    by_age = {
        r["age_group"]: r for r in payload["social_security_beneficiaries"]["by_age"]
    }
    assert by_age["all"]["weighted_beneficiaries"] == 10
    assert by_age["65_plus"]["weighted_beneficiaries"] == 5
    assert by_age["under_65"]["weighted_beneficiaries"] == 5
    assert by_age["all"]["relative_difference"] == pytest.approx(10 / 68_055_000 - 1)
    by_component = {
        r["component"]: r
        for r in payload["social_security_beneficiaries"]["by_component"]
    }
    assert (
        by_component["social_security_retirement"]["weighted_people_with_component"]
        == 7
    )
    assert (
        by_component["social_security_retirement"][
            "weighted_people_with_only_this_component"
        ]
        == 2
    )
    assert (
        by_component["social_security_survivors"]["benchmark_beneficiaries"]
        == 5_786_000
    )

    structure = payload["social_security_component_structure"]
    assert structure["records_with_multiple_components"] == 1
    assert structure["weighted_people_with_multiple_components"] == 5
    assert structure["share_of_beneficiaries_with_multiple_components"] == 0.5
    assert structure["people_by_positive_component_count"]["4"]["records"] == 1

    assert payload["enforced"] is False
    assert payload["used_for_calibration"] is False
    assert payload["target_role"] == "validation"
    assert payload["release_id"] == "fixture"
    assert "gate" not in payload
    assert "December 2024" in payload["comparability"]
    path = write_population_structure_holdout(payload, tmp_path / "diagnostic.json")
    assert json.loads(path.read_text()) == payload


def test_without_a_maritl_the_measure_falls_back_to_unit_structure():
    fixture = _fixture()
    fixture.pop("marital_code")
    payload = population_structure_holdout_payload(
        **fixture,
        is_surviving_spouse=np.array([False, False, True, False, False, False]),
        is_separated=np.zeros(6, dtype=bool),
        period=2024,
    )
    section = payload["marital_status_65_plus"]
    assert "marital-unit" in section["measure"]
    married = _status(payload, "all", "married_spouse_present")
    assert married["share"] == pytest.approx(4 / 12)
    assert _status(payload, "all", "widowed")["share"] == pytest.approx(3 / 12)
    lumped = _status(payload, "all", "other_unmarried_or_spouse_absent")
    assert lumped["share"] == pytest.approx(5 / 12)
    # Without A_MARITL the model can only see a resident spouse, so the
    # benchmark's spouse-absent married people move to the lumped row.
    acs = ACS_2024_B12002_65_PLUS["male"] + ACS_2024_B12002_65_PLUS["female"]
    assert married["benchmark_share"] == pytest.approx(
        acs.married_spouse_present / acs.total
    )
    assert lumped["benchmark_share"] == pytest.approx(
        (acs.divorced + acs.never_married + acs.married_spouse_absent_other) / acs.total
    )


def test_fallback_flags_treat_missing_values_as_false():
    fixture = _fixture()
    fixture.pop("marital_code")
    payload = population_structure_holdout_payload(
        **fixture,
        is_surviving_spouse=np.array([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan]),
        is_separated=None,
        period=2024,
    )
    assert _status(payload, "all", "widowed")["share"] == pytest.approx(3 / 12)


def test_unknown_marital_codes_are_reported_and_excluded_from_shares():
    fixture = _fixture()
    # The divorced man (66, weight 5) carries a non-CPS code.
    fixture["marital_code"] = np.array([1, 1, 4, np.nan, 7, 7])
    payload = population_structure_holdout_payload(**fixture, period=2024)
    row = next(
        r for r in payload["marital_status_65_plus"]["by_sex"] if r["sex"] == "all"
    )
    assert row["weighted_people_unknown_status"] == 5
    assert row["weighted_people"] == 7
    assert _status(payload, "all", "married")["share"] == pytest.approx(4 / 7)
    # The other sections do not depend on marital status.
    assert payload["living_alone_65_plus"]["share"] == pytest.approx(3 / 12)


def test_misaligned_or_invalid_arrays_refuse():
    fixture = _fixture()
    fixture["weights"] = fixture["weights"][:-1]
    with pytest.raises(ValueError, match="aligned"):
        population_structure_holdout_payload(**fixture, period=2024)
    fixture = _fixture()
    fixture["weights"] = -fixture["weights"]
    with pytest.raises(ValueError, match="nonnegative"):
        population_structure_holdout_payload(**fixture, period=2024)
    fixture = _fixture()
    del fixture["components"]["social_security_dependents"]
    with pytest.raises(ValueError, match="component"):
        population_structure_holdout_payload(**fixture, period=2024)


def test_no_weighted_older_people_reports_null_shares():
    fixture = _fixture()
    fixture["weights"] = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    payload = population_structure_holdout_payload(**fixture, period=2025)
    for row in payload["marital_status_65_plus"]["by_sex"]:
        assert row["weighted_people"] == 0
        assert all(s["share"] is None for s in row["statuses"])
        assert all(s["difference_percentage_points"] is None for s in row["statuses"])
    assert payload["living_alone_65_plus"]["share"] is None


def test_h5_reader_matches_the_array_payload(tmp_path):
    fixture = _fixture()
    person = pd.DataFrame(
        {
            "person_id": np.arange(6),
            "person_household_id": fixture["household_id"],
            "person_marital_unit_id": fixture["marital_unit_id"],
            "age": fixture["age"].astype(float),
            "is_female": fixture["is_female"],
            "A_MARITL": fixture["marital_code"],
            **fixture["components"],
        }
    )
    household = pd.DataFrame(
        {"household_id": [1, 2, 3, 4], "household_weight": [2.0, 3.0, 5.0, 7.0]}
    )
    path = tmp_path / "release.h5"
    with pd.HDFStore(path, "w") as store:
        store["person"] = person
        store["household"] = household
    from_h5 = population_structure_holdout_from_h5(path, period=2024, release_id="h5")
    from_arrays = population_structure_holdout_payload(
        **fixture, period=2024, release_id="h5"
    )
    assert from_h5 == from_arrays


def test_unavailable_receipt_is_observational():
    payload = population_structure_holdout_unavailable(
        OSError("missing"), period=2024, release_id="r"
    )
    assert payload["status"] == "unavailable"
    assert payload["enforced"] is False
    assert payload["used_for_calibration"] is False
    assert payload["error"] == "OSError: missing"


@st.composite
def populations(draw):
    n_households = draw(st.integers(min_value=1, max_value=12))
    sizes = draw(
        st.lists(st.integers(1, 4), min_size=n_households, max_size=n_households)
    )
    household_id = np.repeat(np.arange(n_households), sizes)
    n = len(household_id)
    age = np.array(draw(st.lists(st.integers(0, 95), min_size=n, max_size=n)))
    marital_unit_id = np.array(
        [10 * h + draw(st.integers(0, 2)) for h in household_id], dtype=np.int64
    )
    marital_code = np.array(draw(st.lists(st.integers(1, 7), min_size=n, max_size=n)))
    is_female = np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)))
    household_weight = np.array(
        draw(
            st.lists(
                st.one_of(st.just(0.0), st.floats(1e-3, 1e4)),
                min_size=n_households,
                max_size=n_households,
            )
        )
    )
    components = {
        name: np.array(
            draw(
                st.lists(
                    st.sampled_from([0.0, 0.0, 0.0, 1.0, 1_000.0, 25_000.0]),
                    min_size=n,
                    max_size=n,
                )
            )
        )
        for name in SOCIAL_SECURITY_COMPONENTS
    }
    return dict(
        age=age,
        is_female=is_female,
        household_id=household_id,
        marital_unit_id=marital_unit_id,
        marital_code=marital_code,
        components=components,
        weights=household_weight[household_id],
    )


def _numbers(payload):
    """Every share and weighted count, for invariance comparisons."""
    out = {}
    for row in payload["marital_status_65_plus"]["by_sex"]:
        out[f"total.{row['sex']}"] = row["weighted_people"]
        for status in row["statuses"]:
            out[f"share.{row['sex']}.{status['status']}"] = status["share"]
    out["alone"] = payload["living_alone_65_plus"]["share"]
    for row in payload["social_security_beneficiaries"]["by_age"]:
        out[f"ben.{row['age_group']}"] = row["weighted_beneficiaries"]
    for row in payload["social_security_beneficiaries"]["by_component"]:
        out[f"comp.{row['component']}"] = row["weighted_people_with_component"]
    structure = payload["social_security_component_structure"]
    out["multiple"] = structure["weighted_people_with_multiple_components"]
    return out


def _close(a, b, scale=1.0):
    for key in a:
        if a[key] is None or b[key] is None:
            assert a[key] is None and b[key] is None, key
        elif key.startswith(("share", "alone")):
            assert b[key] == pytest.approx(a[key], rel=1e-9, abs=1e-12), key
        else:
            assert b[key] == pytest.approx(scale * a[key], rel=1e-9, abs=1e-9), key


@settings(max_examples=100, deadline=None)
@given(populations(), st.floats(0.01, 1e3))
def test_shares_are_invariant_to_weight_scale(population, factor):
    base = _numbers(population_structure_holdout_payload(**population, period=2024))
    scaled = dict(population, weights=population["weights"] * factor)
    _close(
        base,
        _numbers(population_structure_holdout_payload(**scaled, period=2024)),
        factor,
    )


@settings(max_examples=100, deadline=None)
@given(populations(), st.randoms(use_true_random=False))
def test_results_are_invariant_to_person_order(population, rng):
    order = np.arange(len(population["age"]))
    rng.shuffle(order)
    shuffled = {
        key: (
            {name: values[order] for name, values in value.items()}
            if key == "components"
            else value[order]
        )
        for key, value in population.items()
    }
    _close(
        _numbers(population_structure_holdout_payload(**population, period=2024)),
        _numbers(population_structure_holdout_payload(**shuffled, period=2024)),
    )


@settings(max_examples=100, deadline=None)
@given(populations())
def test_accounting_identities(population):
    payload = population_structure_holdout_payload(**population, period=2024)
    weights = population["weights"]
    for row in payload["marital_status_65_plus"]["by_sex"]:
        shares = [s["share"] for s in row["statuses"]]
        if row["weighted_people"] > 0:
            assert sum(shares) == pytest.approx(1.0)
            assert all(-1e-12 <= s <= 1.0 + 1e-12 for s in shares)
        else:
            assert all(s is None for s in shares)
    sexes = {
        r["sex"]: r["weighted_people"]
        for r in payload["marital_status_65_plus"]["by_sex"]
    }
    assert sexes["male"] + sexes["female"] == pytest.approx(sexes["all"])
    by_age = {
        r["age_group"]: r["weighted_beneficiaries"]
        for r in payload["social_security_beneficiaries"]["by_age"]
    }
    assert by_age["65_plus"] + by_age["under_65"] == pytest.approx(by_age["all"])
    counts = payload["social_security_component_structure"][
        "people_by_positive_component_count"
    ]
    assert sum(c["weighted_people"] for c in counts.values()) == pytest.approx(
        weights.sum()
    )
    assert sum(c["records"] for c in counts.values()) == len(weights)
    beneficiaries = sum(c["weighted_people"] for k, c in counts.items() if k != "0")
    assert beneficiaries == pytest.approx(by_age["all"])
    alone = payload["living_alone_65_plus"]["share"]
    assert alone is None or -1e-12 <= alone <= 1.0 + 1e-12


@settings(max_examples=50, deadline=None)
@given(populations())
def test_single_component_populations_report_no_multiple_components(population):
    stacked = np.column_stack(
        [population["components"][c] for c in SOCIAL_SECURITY_COMPONENTS]
    )
    keep_first = np.zeros_like(stacked)
    first = np.argmax(stacked > 0, axis=1)
    rows = np.arange(len(stacked))
    keep_first[rows, first] = stacked[rows, first]
    population = dict(
        population,
        components={
            c: keep_first[:, i] for i, c in enumerate(SOCIAL_SECURITY_COMPONENTS)
        },
    )
    structure = population_structure_holdout_payload(**population, period=2024)[
        "social_security_component_structure"
    ]
    assert structure["records_with_multiple_components"] == 0
    assert structure["weighted_people_with_multiple_components"] == 0
