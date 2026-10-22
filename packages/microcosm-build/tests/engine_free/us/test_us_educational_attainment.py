"""The auxiliary CPS educational-attainment column and its S1501 targets (#297)."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import educational_attainment as ea
from microcosm.build.us_runtime.fiscal_targets import (
    US_EDUCATIONAL_ATTAINMENT_TARGET_ROLE,
    compile_us_fiscal_target_registry,
)
from microcosm.build.us_runtime.spm_role_source import AsecSpmRoleSource
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.units import US_SCHEMA
from test_support.microcosm_build.us_fiscal_targets import (
    _dynamic_ledger_fact,
    packaged_reference_facts,
)
from test_support.microcosm_build.us_stored_input_register import (
    _load_builder_module,
)

# ---------------------------------------------------------------------------
# Code maps, checked against the Census dictionaries
# ---------------------------------------------------------------------------


def test_a_hga_map_covers_exactly_the_census_codes():
    """ASEC 2023-2026 data dictionaries, Item 18h: 0 (Children) and 31-46."""

    assert set(ea.ASEC_A_HGA_ATTAINMENT_CLASSES) == {0, *range(31, 47)}
    expected = {
        ea.NOT_IN_UNIVERSE: [0],
        ea.LESS_THAN_9TH_GRADE: [31, 32, 33, 34],
        ea.GRADE_9_TO_12_NO_DIPLOMA: [35, 36, 37, 38],
        ea.HIGH_SCHOOL_GRADUATE: [39],
        ea.SOME_COLLEGE_NO_DEGREE: [40],
        ea.ASSOCIATES_DEGREE: [41, 42],
        ea.BACHELORS_DEGREE: [43],
        ea.GRADUATE_OR_PROFESSIONAL_DEGREE: [44, 45, 46],
    }
    for label, codes in expected.items():
        assert [
            code
            for code, mapped in ea.ASEC_A_HGA_ATTAINMENT_CLASSES.items()
            if mapped == label
        ] == codes


def test_schl_map_covers_exactly_the_pums_codes():
    """2024 ACS PUMS data dictionary, SCHL: 01-24, blank under age 3."""

    assert set(ea.ACS_SCHL_ATTAINMENT_CLASSES) == set(range(1, 25))
    assert ea.ACS_SCHL_ATTAINMENT_CLASSES[11] == ea.LESS_THAN_9TH_GRADE
    assert ea.ACS_SCHL_ATTAINMENT_CLASSES[15] == ea.GRADE_9_TO_12_NO_DIPLOMA
    assert ea.ACS_SCHL_ATTAINMENT_CLASSES[17] == ea.HIGH_SCHOOL_GRADUATE
    assert ea.ACS_SCHL_ATTAINMENT_CLASSES[19] == ea.SOME_COLLEGE_NO_DEGREE
    assert ea.ACS_SCHL_ATTAINMENT_CLASSES[20] == ea.ASSOCIATES_DEGREE
    assert ea.ACS_SCHL_ATTAINMENT_CLASSES[21] == ea.BACHELORS_DEGREE
    assert ea.ACS_SCHL_ATTAINMENT_CLASSES[23] == ea.GRADUATE_OR_PROFESSIONAL_DEGREE
    assert set(ea.ACS_SCHL_ATTAINMENT_CLASSES.values()) == set(
        ea.US_EDUCATIONAL_ATTAINMENT_CLASSES
    )


def test_both_sources_map_onto_the_same_classes():
    a_hga = set(ea.ASEC_A_HGA_ATTAINMENT_CLASSES.values())
    assert a_hga == set(ea.US_EDUCATIONAL_ATTAINMENT_VALUES)
    assert set(ea.ACS_SCHL_ATTAINMENT_CLASSES.values()) | {ea.NOT_IN_UNIVERSE} == a_hga


def test_a_hga_refuses_codes_the_dictionary_does_not_list():
    assert list(ea.attainment_from_a_hga([0, 39, 42, 46])) == [
        ea.NOT_IN_UNIVERSE,
        ea.HIGH_SCHOOL_GRADUATE,
        ea.ASSOCIATES_DEGREE,
        ea.GRADUATE_OR_PROFESSIONAL_DEGREE,
    ]
    for bad in ([30], [47], [-1], [np.nan], [39.5]):
        with pytest.raises(ea.EducationalAttainmentError):
            ea.attainment_from_a_hga(bad)


def test_schl_puts_everyone_under_15_out_of_universe():
    classes = ea.attainment_from_schl(["", "03", "16", "21", "13"], [2, 5, 30, 40, 14])
    assert list(classes) == [
        ea.NOT_IN_UNIVERSE,
        ea.NOT_IN_UNIVERSE,
        ea.HIGH_SCHOOL_GRADUATE,
        ea.BACHELORS_DEGREE,
        ea.NOT_IN_UNIVERSE,
    ]
    with pytest.raises(ea.EducationalAttainmentError, match="blank"):
        ea.attainment_from_schl([""], [3])
    with pytest.raises(ea.EducationalAttainmentError):
        ea.attainment_from_schl(["25"], [40])


def test_the_six_requested_classes_and_the_roi_classes():
    assert sorted(set(ea.US_EDUCATIONAL_ATTAINMENT_SIX_CLASSES.values())) == sorted(
        {
            "less_than_high_school",
            ea.HIGH_SCHOOL_GRADUATE,
            ea.SOME_COLLEGE_NO_DEGREE,
            ea.ASSOCIATES_DEGREE,
            ea.BACHELORS_DEGREE,
            ea.GRADUATE_OR_PROFESSIONAL_DEGREE,
            ea.NOT_IN_UNIVERSE,
        }
    )
    assert dict(ea.US_DEGREE_ROI_CLASSES) == {
        "baseline": ea.HIGH_SCHOOL_GRADUATE,
        "two_year_degree": ea.ASSOCIATES_DEGREE,
        "four_year_degree": ea.BACHELORS_DEGREE,
    }


def test_s1501_categories_partition_the_classes_in_each_universe():
    categories = ea.S1501_ATTAINMENT_CATEGORY_CLASSES
    over_25 = [categories[label] for label in ea.US_EDUCATIONAL_ATTAINMENT_CLASSES]
    assert all(len(classes) == 1 for classes in over_25)
    under_25 = [
        categories["less_than_high_school_graduate"],
        categories[ea.HIGH_SCHOOL_GRADUATE],
        categories["some_college_or_associates_degree"],
        categories["bachelors_degree_or_higher"],
    ]
    for universe in (over_25, under_25):
        union = set().union(*universe)
        assert union == set(ea.US_EDUCATIONAL_ATTAINMENT_CLASSES)
        assert sum(len(classes) for classes in universe) == len(union)


# ---------------------------------------------------------------------------
# The exact Census join
# ---------------------------------------------------------------------------


def _peridnum(index: int) -> str:
    return f"{index:022d}"


def _member(year: int) -> pd.DataFrame:
    """One synthetic Census person member: three people in one household."""

    return pd.DataFrame(
        {
            "PERIDNUM": [_peridnum(year * 10 + line) for line in (1, 2, 3)],
            "PH_SEQ": [7, 7, 7],
            "P_SEQ": [1, 2, 3],
            "A_LINENO": [1, 2, 3],
            "A_AGE": [44, 41, 9],
            "A_HGA": [43, 41, 0],
        }
    )


def _pin(path, year: int, rows: int) -> AsecSpmRoleSource:
    data = path.read_bytes()
    return AsecSpmRoleSource(
        income_year=year,
        survey_year=year + 1,
        csv_sha256=hashlib.sha256(data).hexdigest(),
        csv_size_bytes=len(data),
        persons=rows,
        units=1,
        official_archive_url="https://example.org/asec.zip",
        archive_sha256="0" * 64,
        member=f"pppub{str(year + 1)[2:]}.csv",
    )


def _write_member(tmp_path, year: int, member: pd.DataFrame | None = None):
    member = _member(year) if member is None else member
    path = tmp_path / f"pppub{str(year + 1)[2:]}.csv"
    member.to_csv(path, index=False)
    return path, _pin(path, year, len(member))


def _sources(tmp_path, years=(2023, 2024)) -> pd.DataFrame:
    paths, pins = {}, {}
    for year in years:
        paths[year], pins[year] = _write_member(tmp_path, year)
    return ea.load_asec_attainment_sources(years, paths, pins=pins)


def _person(year: int, *, clone: bool = True) -> pd.DataFrame:
    """The member's people as a pooled frame stores them, plus a PUF clone."""

    rows = pd.DataFrame(
        {
            "source_year": year,
            "PERIDNUM": [_peridnum(year * 10 + line) for line in (1, 2, 3)],
            # Pooling globalizes PH_SEQ; source_household_id keeps Census's.
            "PH_SEQ": [900, 900, 900],
            "source_household_id": [7, 7, 7],
            "P_SEQ": [1, 2, 3],
            "A_LINENO": [1, 2, 3],
            "A_AGE": [44, 41, 9],
            "age": [44.0, 41.0, 9.0],
        }
    )
    if clone:
        rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    return rows


def test_every_person_takes_its_own_members_code(tmp_path):
    sources = _sources(tmp_path)
    person = pd.concat([_person(2023), _person(2024, clone=False)], ignore_index=True)

    a_hga, classes, record = ea.derive_cps_educational_attainment(person, sources)

    assert a_hga.tolist() == [43, 41, 0, 43, 43, 41, 0]
    assert classes.tolist() == [
        ea.BACHELORS_DEGREE,
        ea.ASSOCIATES_DEGREE,
        ea.NOT_IN_UNIVERSE,
        ea.BACHELORS_DEGREE,
        ea.BACHELORS_DEGREE,
        ea.ASSOCIATES_DEGREE,
        ea.NOT_IN_UNIVERSE,
    ]
    assert record["column_role"] == "auxiliary"
    assert record["joined_persons_by_income_year"] == {"2023": 4, "2024": 3}
    assert set(record["identity_columns_verified"]) == {
        "source_household_id",
        "P_SEQ",
        "A_LINENO",
        "A_AGE",
    }
    assert record["sources"]["2023"]["a_hga_counts"]["43"] == 1


def test_the_join_refuses_a_person_whose_identity_disagrees(tmp_path):
    sources = _sources(tmp_path)
    person = _person(2023)
    person.loc[1, "A_LINENO"] = 5
    with pytest.raises(ea.EducationalAttainmentError, match="A_LINENO"):
        ea.derive_cps_educational_attainment(person, sources)
    person = _person(2023)
    person.loc[0, "source_household_id"] = 8
    with pytest.raises(ea.EducationalAttainmentError, match="source_household_id"):
        ea.derive_cps_educational_attainment(person, sources)


def test_the_join_checks_raw_ph_seq_on_an_unpooled_table(tmp_path):
    sources = _sources(tmp_path)
    person = _person(2023).drop(columns="source_household_id")
    with pytest.raises(ea.EducationalAttainmentError, match="PH_SEQ"):
        ea.derive_cps_educational_attainment(person, sources)
    person["PH_SEQ"] = 7
    ea.derive_cps_educational_attainment(person, sources)


def test_the_join_refuses_acs_rows_unknown_keys_and_uncovered_years(tmp_path):
    sources = _sources(tmp_path)
    acs = _person(2023)
    acs.loc[2, "PERIDNUM"] = None
    with pytest.raises(ea.EducationalAttainmentError, match="SCHL"):
        ea.derive_cps_educational_attainment(acs, sources)
    unknown = _person(2023)
    unknown.loc[2, "PERIDNUM"] = _peridnum(1)
    with pytest.raises(ea.EducationalAttainmentError, match="no row"):
        ea.derive_cps_educational_attainment(unknown, sources)
    with pytest.raises(ea.EducationalAttainmentError, match="2022"):
        ea.derive_cps_educational_attainment(_person(2022), sources)


def test_the_loader_refuses_a_member_off_its_pin_or_universe(tmp_path):
    path, pin = _write_member(tmp_path, 2023)
    short = dataclasses.replace(pin, persons=4)
    with pytest.raises(ea.EducationalAttainmentError, match="persons"):
        ea.load_asec_attainment_sources([2023], {2023: path}, pins={2023: short})
    stale = dataclasses.replace(pin, csv_sha256="1" * 64)
    with pytest.raises(ea.EducationalAttainmentError):
        ea.load_asec_attainment_sources([2023], {2023: path}, pins={2023: stale})

    adult_zero = _member(2023)
    adult_zero.loc[1, "A_HGA"] = 0
    path, pin = _write_member(tmp_path, 2023, adult_zero)
    with pytest.raises(ea.EducationalAttainmentError, match="15 and over"):
        ea.load_asec_attainment_sources([2023], {2023: path}, pins={2023: pin})

    unlisted = _member(2023)
    unlisted.loc[0, "A_HGA"] = 47
    path, pin = _write_member(tmp_path, 2023, unlisted)
    with pytest.raises(ea.EducationalAttainmentError, match="reviewed Census"):
        ea.load_asec_attainment_sources([2023], {2023: path}, pins={2023: pin})
    with pytest.raises(ea.EducationalAttainmentError, match="income year 2031"):
        ea.load_asec_attainment_sources([2031])


# ---------------------------------------------------------------------------
# The frame stage and its gate
# ---------------------------------------------------------------------------


def _frame(person: pd.DataFrame, weights=None) -> Frame:
    n = len(person)
    ids = np.arange(1, n + 1, dtype=np.int64)
    person = person.copy()
    person["person_id"] = ids
    person["person_household_id"] = ids
    for entity, factor in (
        ("tax_unit", 10),
        ("spm_unit", 100),
        ("family", 1_000),
        ("marital_unit", 10_000),
    ):
        person[f"person_{entity}_id"] = ids * factor
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": ids, "state_fips": np.full(n, 6, dtype=np.int64)}
        ),
        **{
            entity: pd.DataFrame({f"{entity}_id": ids * factor})
            for entity, factor in (
                ("tax_unit", 10),
                ("spm_unit", 100),
                ("family", 1_000),
                ("marital_unit", 10_000),
            )
        },
    }
    household_weights = (
        np.full(n, 1_000.0) if weights is None else np.asarray(weights, dtype=float)
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(household_weights, WeightKind.CALIBRATED)},
    )


def _realistic_person() -> pd.DataFrame:
    """Ten adults 25+ in the national mix: one below a diploma, four BA+."""

    codes = [37, 39, 39, 40, 41, 43, 43, 44, 46, 39]
    return pd.DataFrame(
        {
            "source_year": 2023,
            "PERIDNUM": [_peridnum(500 + index) for index in range(10)],
            "source_household_id": 5,
            "P_SEQ": range(1, 11),
            "A_LINENO": range(1, 11),
            "A_AGE": 40,
            "age": 40.0,
            "A_HGA": codes,
        }
    )


def _realistic_sources(tmp_path, person: pd.DataFrame) -> pd.DataFrame:
    member = person.drop(columns=["source_year", "age", "source_household_id"])
    member.insert(1, "PH_SEQ", 5)
    path, pin = _write_member(tmp_path, 2023, member)
    return ea.load_asec_attainment_sources([2023], {2023: path}, pins={2023: pin})


def test_the_stage_appends_both_columns_and_passes_its_gate(tmp_path):
    person = _realistic_person()
    sources = _realistic_sources(tmp_path, person)
    frame = _frame(person.drop(columns="A_HGA"))

    staged, record = ea.with_us_cps_educational_attainment(frame, sources=sources)

    out = staged.table("person")
    assert out["A_HGA"].tolist() == person["A_HGA"].tolist()
    assert out[ea.US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN].tolist()[:2] == [
        ea.GRADE_9_TO_12_NO_DIPLOMA,
        ea.HIGH_SCHOOL_GRADUATE,
    ]
    assert record["columns_added"] == ["A_HGA", ea.US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN]
    gate = ea.us_cps_educational_attainment_signal_gate(staged)
    assert gate.passed, gate.failures
    assert gate.details["bachelors_or_higher_share_25_plus"] == pytest.approx(0.4)
    assert gate.details["less_than_high_school_share_25_plus"] == pytest.approx(0.1)


def test_the_stage_never_overwrites_a_differing_column(tmp_path):
    person = _realistic_person()
    sources = _realistic_sources(tmp_path, person)
    carried = person.copy()
    staged, record = ea.with_us_cps_educational_attainment(
        _frame(carried), sources=sources
    )
    assert record["columns_verified_equal"] == ["A_HGA"]
    carried.loc[0, "A_HGA"] = 39
    with pytest.raises(ea.EducationalAttainmentError, match="never overwritten"):
        ea.with_us_cps_educational_attainment(_frame(carried), sources=sources)


def test_the_gate_refuses_an_implausible_mix(tmp_path):
    person = _realistic_person()
    person["A_HGA"] = 43
    sources = _realistic_sources(tmp_path, person)
    staged, _ = ea.with_us_cps_educational_attainment(
        _frame(person.drop(columns="A_HGA")), sources=sources
    )
    gate = ea.us_cps_educational_attainment_signal_gate(staged)
    assert not gate.passed
    assert any("bachelors_or_higher" in failure for failure in gate.failures)


# ---------------------------------------------------------------------------
# The S1501 compile path
# ---------------------------------------------------------------------------


def _s1501_fact(
    *,
    category: str,
    value: float,
    universe: str = "25_plus",
    fips: str = "06",
    age_constraints: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    record_set_id = f"census_acs.acs1_2024.s1501.state_attainment_{universe}.{fips}"
    if age_constraints is None:
        age_constraints = (
            [{"variable": "age", "operator": ">=", "value": 25, "role": "filter"}]
            if universe == "25_plus"
            else [
                {"variable": "age", "operator": ">=", "value": 18, "role": "filter"},
                {"variable": "age", "operator": "<", "value": 25, "role": "filter"},
            ]
        )
    constraints = list(age_constraints)
    if category != "all":
        constraints.insert(
            0,
            {
                "variable": "educational_attainment",
                "operator": "==",
                "value": category,
                "role": "filter",
            },
        )
    return _dynamic_ledger_fact(
        source_record_id=f"{record_set_id}.{category}.population",
        source_name="census_acs",
        measure_id="population",
        value=value,
        period_value=2024,
        geography_level="state",
        geography_id=f"0400000US{fips}",
        groupby_dimension="educational_attainment",
        groupby_value_id=category,
        layout_record_set_id=record_set_id,
        universe_constraints=constraints,
    )


def _compile(*facts):
    return compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *facts], allow_unaged_dollar_targets=True
    )


def test_only_state_25_plus_detail_rows_bind():
    registry = _compile(
        _s1501_fact(category=ea.BACHELORS_DEGREE, value=6_355_683),
        _s1501_fact(category=ea.ASSOCIATES_DEGREE, value=2_200_000),
        _s1501_fact(category="all", value=27_000_000),
        _s1501_fact(
            category="bachelors_degree_or_higher", value=1, universe="18_to_24"
        ),
        _s1501_fact(category=ea.BACHELORS_DEGREE, value=5, fips="72"),
    )
    attainment = [
        spec
        for spec in registry.specs
        if spec.family == ea.US_EDUCATIONAL_ATTAINMENT_TARGET_FAMILY
    ]
    assert sorted(spec.value for spec in attainment) == [2_200_000, 6_355_683]
    bachelors = next(spec for spec in attainment if spec.value == 6_355_683)
    metadata = bachelors.metadata
    assert metadata["materializer"] == ea.US_EDUCATIONAL_ATTAINMENT_TARGET_MATERIALIZER
    assert metadata["target_role"] == US_EDUCATIONAL_ATTAINMENT_TARGET_ROLE
    assert metadata["measure_mode"] == "indicator_sum"
    assert metadata["state_fips"] == "06"
    assert metadata["age_lower_bound"] == "25"
    assert metadata["age_upper_bound"] == "inf"
    assert metadata[ea.US_EDUCATIONAL_ATTAINMENT_CLASSES_METADATA_KEY] == (
        ea.BACHELORS_DEGREE
    )
    assert bachelors.name == (
        "census_acs.acs1_2024.s1501.state_attainment_25_plus.06."
        "bachelors_degree.population"
    )
    names = {spec.name for spec in attainment}
    assert {spec.name for spec in registry.specs if "s1501" in spec.name} == names


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"category": "doctorate_only"}, "unreviewed attainment category"),
        (
            {
                "category": ea.BACHELORS_DEGREE,
                "age_constraints": [
                    {"variable": "age", "operator": ">=", "value": 18, "role": "filter"}
                ],
            },
            "not the 25_plus universe",
        ),
    ],
)
def test_a_changed_package_raises_instead_of_binding_wrong(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _compile(_s1501_fact(value=1, **kwargs))


# ---------------------------------------------------------------------------
# The release tool's materializer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def builder():
    return _load_builder_module()


def _attainment_spec():
    registry = _compile(_s1501_fact(category=ea.ASSOCIATES_DEGREE, value=2_200_000))
    return next(
        spec
        for spec in registry.specs
        if spec.family == ea.US_EDUCATIONAL_ATTAINMENT_TARGET_FAMILY
    )


def test_the_tool_accepts_the_compiled_age_restatement(builder):
    spec = _attainment_spec()
    assert spec.metadata["ledger_filter_age_lower_bound"] == "25"
    assert builder._unsupported_ledger_filter_metadata([spec]) == {}


def test_the_tool_counts_persons_in_class_band_and_state(builder):
    spec = _attainment_spec()
    person = pd.DataFrame(
        {
            ea.US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN: [
                ea.ASSOCIATES_DEGREE,
                ea.ASSOCIATES_DEGREE,
                ea.BACHELORS_DEGREE,
                ea.ASSOCIATES_DEGREE,
            ],
            "age": [30.0, 24.0, 50.0, 60.0],
        }
    )
    frame = _frame(person)
    household = frame.table("household").copy()
    household.loc[3, "state_fips"] = 36
    values = builder._population_attainment_household_values(
        frame=frame,
        household=household,
        age=person["age"].to_numpy(),
        metadata=spec.metadata,
    )
    assert values.tolist() == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="--cps-educational-attainment"):
        builder._population_attainment_household_values(
            frame=_frame(person.drop(columns=ea.US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN)),
            household=household,
            age=person["age"].to_numpy(),
            metadata=spec.metadata,
        )


def test_the_tool_parses_attainment_sources(builder):
    assert builder._asec_attainment_source_paths(["2024=a.zip", "2023=b.csv"]) == {
        2024: Path("a.zip"),
        2023: Path("b.csv"),
    }
    for bad in (["2024"], ["x=a.zip"], ["2024=a", "2024=b"]):
        with pytest.raises(SystemExit):
            builder._asec_attainment_source_paths(bad)
    args = builder._parse_args(
        [
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--cps-educational-attainment",
            "--asec-attainment-source",
            "2024=a.zip",
        ]
    )
    assert args.cps_educational_attainment
    assert args.asec_attainment_source == ["2024=a.zip"]
