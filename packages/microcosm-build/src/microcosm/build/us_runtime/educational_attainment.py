"""Educational attainment as an auxiliary person column (microcosm#297).

The student debt pilot's state degree-ROI work (PolicyEngine/us-student-
debt#1, Project 2) reruns PolicyEngine with counterfactual earnings for each
degree holder, so it needs every person's highest completed level next to the
engine outputs for that person. policyengine-us has no attainment input, and
none is needed: the column is *auxiliary*. Microcosm carries it for
conditioning (imputation predictors) and calibration (ACS S1501 state
margins), and no rule consumes it. The stored-input register says so
(:data:`microcosm.data.stored_inputs.US_AUXILIARY_STORED_COLUMNS`), and the
analysis joins it on ``person_id``.

**Classes.** Seven, the population-25-and-over partition of ACS subject table
S1501 (Educational Attainment), whose Chronicle value ids they reuse, plus
``not_in_universe`` for people under 15:

======================================  ==================  ===================
Class                                   CPS ASEC ``A_HGA``  ACS PUMS ``SCHL``
======================================  ==================  ===================
``less_than_9th_grade``                 31-34               01-11
``9th_to_12th_grade_no_diploma``        35-38               12-15
``high_school_graduate_includes_        39                  16, 17
equivalency``
``some_college_no_degree``              40                  18, 19
``associates_degree``                   41, 42              20
``bachelors_degree``                    43                  21
``graduate_or_professional_degree``     44, 45, 46          22, 23, 24
``not_in_universe``                     0 ("Children")      blank (under 3), or
                                                            any person under 15
======================================  ==================  ===================

Collapsing the first two gives the six classes microcosm#297 asks for (less
than high school; high school diploma or GED; some college, no degree;
associate's, occupational and academic together; bachelor's; graduate or
professional): :data:`US_EDUCATIONAL_ATTAINMENT_SIX_CLASSES`. The degree-ROI
analysis compares :data:`US_DEGREE_ROI_CLASSES`: a two-year degree is
``associates_degree`` and a four-year degree ``bachelors_degree``, each against
the ``high_school_graduate_includes_equivalency`` baseline.

**Sources.** ``A_HGA`` ("Item 18h - Educational attainment", universe all
persons, 0 = Children) is read from the official ASEC public-use person
members the build already pins
(:data:`.spm_role_source.ASEC_SPM_ROLE_SOURCES`, the archives of
:data:`.education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES`). The
codes were checked against the Census ASEC public-use data dictionaries for
survey years 2023 to 2026 (``asec<YYYY>_ddl_pub_full.pdf`` under
``https://www2.census.gov/programs-surveys/cps/datasets/<YYYY>/march/``, PDF
pages 22, 23, 24 and 28); the four agree code for code. ``SCHL`` is the ACS
PUMS "Educational attainment" field, checked against
``https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.txt``.
The S1501 categories follow the table's own rows (S1501_C01_006E to 013E for
the population 25 and over; 001E to 005E for 18 to 24).

The CPS column is filled by an exact join on (income year, ``PERIDNUM``) to
the pinned member, with the Census ``PH_SEQ`` (the frame's
``source_household_id``), ``P_SEQ``, ``A_LINENO`` and ``A_AGE`` checked equal
wherever the frame carries them, so a support clone takes its source person's
value. No value is predicted, imputed or defaulted. The ACS
``SCHL`` map is defined and tested here, but the ACS-local lane does not read
``SCHL`` from PUMS yet; ACS-spine rows are refused rather than given a value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

from .education_assistance_source import fetch_asec_education_assistance_source
from .spm_role_source import (
    ASEC_SPM_ROLE_SOURCES,
    AsecSpmRoleSource,
    _exact_person_keys,
    read_pinned_asec_person_columns,
)

__all__ = [
    "ACS_SCHL_ATTAINMENT_CLASSES",
    "ASEC_A_HGA_ATTAINMENT_CLASSES",
    "ASSOCIATES_DEGREE",
    "BACHELORS_DEGREE",
    "GRADE_9_TO_12_NO_DIPLOMA",
    "GRADUATE_OR_PROFESSIONAL_DEGREE",
    "HIGH_SCHOOL_GRADUATE",
    "LESS_THAN_9TH_GRADE",
    "NOT_IN_UNIVERSE",
    "S1501_ATTAINMENT_CATEGORY_CLASSES",
    "S1501_BINDING_UNIVERSE",
    "SOME_COLLEGE_NO_DEGREE",
    "US_ASEC_ATTAINMENT_SOURCE_COLUMN",
    "US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN",
    "US_DEGREE_ROI_CLASSES",
    "US_EDUCATIONAL_ATTAINMENT_CLASSES",
    "US_EDUCATIONAL_ATTAINMENT_CLASSES_METADATA_KEY",
    "US_EDUCATIONAL_ATTAINMENT_MINIMUM_AGE",
    "US_EDUCATIONAL_ATTAINMENT_SIX_CLASSES",
    "US_EDUCATIONAL_ATTAINMENT_TARGET_FAMILY",
    "US_EDUCATIONAL_ATTAINMENT_TARGET_MATERIALIZER",
    "US_EDUCATIONAL_ATTAINMENT_VALUES",
    "EducationalAttainmentError",
    "attainment_classes_from_metadata",
    "attainment_from_a_hga",
    "attainment_from_schl",
    "attainment_target_person_indicator",
    "derive_cps_educational_attainment",
    "load_asec_attainment_sources",
    "us_cps_educational_attainment_signal_gate",
    "us_cps_educational_attainment_summary",
    "with_us_cps_educational_attainment",
]


class EducationalAttainmentError(ValueError):
    """The attainment derivation refused its inputs."""


#: The auxiliary person column, named for its source (CPS ASEC ``A_HGA``).
US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN = "cps_educational_attainment"
#: The raw Census code, carried beside the class (uppercase Census fields pass
#: the stored-input rule by convention; see microcosm.data.stored_inputs).
US_ASEC_ATTAINMENT_SOURCE_COLUMN = "A_HGA"
#: CPS ASEC asks attainment of persons 15 and over; ``A_HGA`` is 0
#: ("Children") below that. The ACS asks from age 3; the column puts every
#: person under 15 out of universe so both sources share one universe.
US_EDUCATIONAL_ATTAINMENT_MINIMUM_AGE = 15

LESS_THAN_9TH_GRADE = "less_than_9th_grade"
GRADE_9_TO_12_NO_DIPLOMA = "9th_to_12th_grade_no_diploma"
HIGH_SCHOOL_GRADUATE = "high_school_graduate_includes_equivalency"
SOME_COLLEGE_NO_DEGREE = "some_college_no_degree"
ASSOCIATES_DEGREE = "associates_degree"
BACHELORS_DEGREE = "bachelors_degree"
GRADUATE_OR_PROFESSIONAL_DEGREE = "graduate_or_professional_degree"
NOT_IN_UNIVERSE = "not_in_universe"

#: The seven classes, lowest to highest: S1501's population-25-and-over rows.
US_EDUCATIONAL_ATTAINMENT_CLASSES: tuple[str, ...] = (
    LESS_THAN_9TH_GRADE,
    GRADE_9_TO_12_NO_DIPLOMA,
    HIGH_SCHOOL_GRADUATE,
    SOME_COLLEGE_NO_DEGREE,
    ASSOCIATES_DEGREE,
    BACHELORS_DEGREE,
    GRADUATE_OR_PROFESSIONAL_DEGREE,
)
#: Every value the column may hold.
US_EDUCATIONAL_ATTAINMENT_VALUES: tuple[str, ...] = (
    *US_EDUCATIONAL_ATTAINMENT_CLASSES,
    NOT_IN_UNIVERSE,
)

#: Class -> the six classes microcosm#297 asks for (the two below-diploma
#: classes merge into ``less_than_high_school``).
US_EDUCATIONAL_ATTAINMENT_SIX_CLASSES: Mapping[str, str] = MappingProxyType(
    {
        LESS_THAN_9TH_GRADE: "less_than_high_school",
        GRADE_9_TO_12_NO_DIPLOMA: "less_than_high_school",
        HIGH_SCHOOL_GRADUATE: HIGH_SCHOOL_GRADUATE,
        SOME_COLLEGE_NO_DEGREE: SOME_COLLEGE_NO_DEGREE,
        ASSOCIATES_DEGREE: ASSOCIATES_DEGREE,
        BACHELORS_DEGREE: BACHELORS_DEGREE,
        GRADUATE_OR_PROFESSIONAL_DEGREE: GRADUATE_OR_PROFESSIONAL_DEGREE,
        NOT_IN_UNIVERSE: NOT_IN_UNIVERSE,
    }
)

#: The classes the state degree-ROI analysis (us-student-debt#5) compares.
US_DEGREE_ROI_CLASSES: Mapping[str, str] = MappingProxyType(
    {
        "baseline": HIGH_SCHOOL_GRADUATE,
        "two_year_degree": ASSOCIATES_DEGREE,
        "four_year_degree": BACHELORS_DEGREE,
    }
)


def _codes(classes: Mapping[str, Iterable[int]]) -> Mapping[int, str]:
    mapped: dict[int, str] = {}
    for label, codes in classes.items():
        for code in codes:
            if code in mapped:
                raise AssertionError(f"code {code} mapped twice")
            mapped[int(code)] = label
    return MappingProxyType(dict(sorted(mapped.items())))


#: ASEC ``A_HGA`` code -> class. The Census dictionaries list exactly these
#: codes (0 and 31 to 46) for survey years 2023 to 2026.
ASEC_A_HGA_ATTAINMENT_CLASSES: Mapping[int, str] = _codes(
    {
        NOT_IN_UNIVERSE: (0,),
        # 31 less than 1st grade; 32 1st-4th; 33 5th-6th; 34 7th and 8th.
        LESS_THAN_9TH_GRADE: range(31, 35),
        # 35 9th; 36 10th; 37 11th; 38 12th grade no diploma.
        GRADE_9_TO_12_NO_DIPLOMA: range(35, 39),
        # 39 high school graduate - high school diploma or equivalent.
        HIGH_SCHOOL_GRADUATE: (39,),
        # 40 some college but no degree.
        SOME_COLLEGE_NO_DEGREE: (40,),
        # 41 associate degree, occupation/vocation program; 42 academic program.
        ASSOCIATES_DEGREE: (41, 42),
        # 43 bachelor's degree.
        BACHELORS_DEGREE: (43,),
        # 44 master's; 45 professional school degree; 46 doctorate.
        GRADUATE_OR_PROFESSIONAL_DEGREE: (44, 45, 46),
    }
)

#: ACS PUMS ``SCHL`` code -> class (2024 PUMS data dictionary). Blank is
#: "N/A (less than 3 years old)".
ACS_SCHL_ATTAINMENT_CLASSES: Mapping[int, str] = _codes(
    {
        # 01 no schooling; 02 nursery; 03 kindergarten; 04-11 grades 1 to 8.
        LESS_THAN_9TH_GRADE: range(1, 12),
        # 12-14 grades 9 to 11; 15 12th grade, no diploma.
        GRADE_9_TO_12_NO_DIPLOMA: range(12, 16),
        # 16 regular high school diploma; 17 GED or alternative credential.
        HIGH_SCHOOL_GRADUATE: (16, 17),
        # 18 some college, less than 1 year; 19 1+ years of credit, no degree.
        SOME_COLLEGE_NO_DEGREE: (18, 19),
        # 20 associate's degree.
        ASSOCIATES_DEGREE: (20,),
        # 21 bachelor's degree.
        BACHELORS_DEGREE: (21,),
        # 22 master's; 23 professional degree beyond a bachelor's; 24 doctorate.
        GRADUATE_OR_PROFESSIONAL_DEGREE: (22, 23, 24),
    }
)

#: S1501 category (the Chronicle ``educational_attainment`` value id of
#: ``packages/census/acs_s1501_state_2024``) -> the classes it counts. The
#: population-25-and-over rows are one class each; the 18-to-24 rows merge
#: neighbouring classes.
S1501_ATTAINMENT_CATEGORY_CLASSES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        **{label: frozenset({label}) for label in US_EDUCATIONAL_ATTAINMENT_CLASSES},
        "less_than_high_school_graduate": frozenset(
            {LESS_THAN_9TH_GRADE, GRADE_9_TO_12_NO_DIPLOMA}
        ),
        "some_college_or_associates_degree": frozenset(
            {SOME_COLLEGE_NO_DEGREE, ASSOCIATES_DEGREE}
        ),
        "bachelors_degree_or_higher": frozenset(
            {BACHELORS_DEGREE, GRADUATE_OR_PROFESSIONAL_DEGREE}
        ),
    }
)
#: The S1501 universe that binds: the population 25 and over. The 18-to-24
#: universe is held out (see
#: :func:`microcosm.build.us_runtime.fiscal_targets._acs_attainment_reference_from_fact`).
S1501_BINDING_UNIVERSE = "25_plus"

#: Target metadata naming the materializer, family and class set.
US_EDUCATIONAL_ATTAINMENT_TARGET_MATERIALIZER = "population_attainment"
US_EDUCATIONAL_ATTAINMENT_TARGET_FAMILY = "census_educational_attainment"
US_EDUCATIONAL_ATTAINMENT_CLASSES_METADATA_KEY = "educational_attainment_classes"

_JOIN_KEY = "PERIDNUM"
_SOURCE_YEAR = "source_year"
_IDENTITY_COLUMNS = ("PH_SEQ", "P_SEQ", "A_LINENO", "A_AGE")
_MEMBER_COLUMNS = (_JOIN_KEY, *_IDENTITY_COLUMNS, US_ASEC_ATTAINMENT_SOURCE_COLUMN)
_EXAMPLES = 5
#: Weighted shares of the population 25 and over the gate requires. ACS 2024
#: S1501 nationally: about 10% below a diploma, 37% bachelor's or higher.
_LESS_THAN_HIGH_SCHOOL_SHARE_BAND = (0.05, 0.16)
_BACHELORS_OR_HIGHER_SHARE_BAND = (0.25, 0.45)


def _refuse(message: str) -> None:
    raise EducationalAttainmentError(message)


def _integer_codes(values: object, *, label: str) -> np.ndarray:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    bad = ~np.isfinite(numeric) | (numeric != np.floor(numeric))
    if bad.any():
        _refuse(
            f"{label} has {int(bad.sum())} missing or non-integer codes, e.g. "
            f"{series[bad].head(_EXAMPLES).tolist()}."
        )
    return numeric.astype(np.int64)


def _classify(codes: np.ndarray, mapping: Mapping[int, str], label: str) -> np.ndarray:
    known = np.isin(codes, np.fromiter(mapping, dtype=np.int64))
    if not known.all():
        _refuse(
            f"{label} has {int((~known).sum())} codes outside the reviewed Census "
            f"codes {sorted(mapping)}, e.g. {sorted(set(codes[~known].tolist()))[:_EXAMPLES]}."
        )
    lookup = pd.Series(mapping)
    return lookup.reindex(codes).to_numpy(dtype=object)


def attainment_from_a_hga(codes: object) -> np.ndarray:
    """Map CPS ASEC ``A_HGA`` codes to classes, refusing any other code."""

    return _classify(
        _integer_codes(codes, label="A_HGA"), ASEC_A_HGA_ATTAINMENT_CLASSES, "A_HGA"
    )


def attainment_from_schl(codes: object, ages: object) -> np.ndarray:
    """Map ACS PUMS ``SCHL`` codes to classes for persons of ``ages``.

    ``SCHL`` is blank only for persons under 3, and every person under
    :data:`US_EDUCATIONAL_ATTAINMENT_MINIMUM_AGE` is ``not_in_universe`` so
    the ACS column shares the CPS universe.
    """

    age = _integer_codes(ages, label="age")
    series = pd.Series(codes).reset_index(drop=True)
    blank = series.isna() | series.astype(str).str.strip().eq("")
    if (blank & pd.Series(age >= 3)).any():
        _refuse("SCHL is blank for a person aged 3 or over.")
    result = np.full(len(series), NOT_IN_UNIVERSE, dtype=object)
    answered = ~blank.to_numpy()
    if answered.any():
        result[answered] = _classify(
            _integer_codes(series[answered], label="SCHL"),
            ACS_SCHL_ATTAINMENT_CLASSES,
            "SCHL",
        )
    result[age < US_EDUCATIONAL_ATTAINMENT_MINIMUM_AGE] = NOT_IN_UNIVERSE
    return result


def _load_one_member(
    path: Path, pin: AsecSpmRoleSource
) -> tuple[pd.DataFrame, dict[str, object]]:
    label = f"ASEC {pin.survey_year} Census person member {pin.member}"
    try:
        source, form = read_pinned_asec_person_columns(
            path, pin, label, _MEMBER_COLUMNS
        )
    except ValueError as error:
        raise EducationalAttainmentError(str(error)) from error
    if len(source) != pin.persons:
        _refuse(f"{label} holds {len(source)} persons; the pin records {pin.persons}.")
    try:
        source[_JOIN_KEY] = _exact_person_keys(source[_JOIN_KEY], label)
    except ValueError as error:
        raise EducationalAttainmentError(str(error)) from error
    if source[_JOIN_KEY].duplicated().any():
        _refuse(f"{label} repeats PERIDNUM; the join must be many-to-one.")
    for column in (*_IDENTITY_COLUMNS, US_ASEC_ATTAINMENT_SOURCE_COLUMN):
        source[column] = _integer_codes(source[column], label=f"{label} {column}")
    codes = source[US_ASEC_ATTAINMENT_SOURCE_COLUMN].to_numpy()
    attainment_from_a_hga(codes)
    child = source["A_AGE"].to_numpy() < US_EDUCATIONAL_ATTAINMENT_MINIMUM_AGE
    zero = codes == 0
    if (zero != child).any():
        _refuse(
            f"{label}: A_HGA is 0 ('Children') for {int((zero & ~child).sum())} "
            f"persons aged 15+ and nonzero for {int((child & ~zero).sum())} under "
            "15; the reviewed universe is persons 15 and over."
        )
    counts = pd.Series(codes).value_counts()
    return source, {
        "income_year": pin.income_year,
        "survey_year": pin.survey_year,
        "source_form": form,
        "official_archive_url": pin.official_archive_url,
        "archive_sha256": pin.archive_sha256,
        "member": pin.member,
        "member_sha256": pin.csv_sha256,
        "persons": len(source),
        "a_hga_counts": {
            str(code): int(counts.get(code, 0))
            for code in ASEC_A_HGA_ATTAINMENT_CLASSES
        },
    }


def load_asec_attainment_sources(
    income_years: Iterable[int],
    paths: Mapping[int, str | Path] | None = None,
    *,
    pins: Mapping[int, AsecSpmRoleSource] | None = None,
) -> pd.DataFrame:
    """Read ``A_HGA`` and the identity columns from each pinned person member.

    ``paths`` maps income years to the official archive or its extracted
    person CSV; an income year without a path is fetched from the Census
    archive and verified (:func:`.education_assistance_source.
    fetch_asec_education_assistance_source`). ``pins`` defaults to
    :data:`.spm_role_source.ASEC_SPM_ROLE_SOURCES`; an explicit mapping is for
    synthetic tests. The result carries ``source_year`` and one row per
    member person; its ``attrs["provenance"]`` records every member read.
    """

    pins = ASEC_SPM_ROLE_SOURCES if pins is None else pins
    years = sorted({int(year) for year in income_years})
    if not years:
        _refuse("No income years to read A_HGA for.")
    parts: list[pd.DataFrame] = []
    provenance: dict[str, object] = {}
    for year in years:
        pin = pins.get(year)
        if pin is None:
            _refuse(
                f"No pinned Census ASEC person member covers income year {year}; "
                f"pinned income years: {sorted(pins)}."
            )
        if pin.income_year != year:
            _refuse(f"Census ASEC pin for {year} is for income year {pin.income_year}.")
        provided = None if paths is None else paths.get(year)
        path = (
            Path(provided).expanduser()
            if provided is not None
            else fetch_asec_education_assistance_source(year)
        )
        source, record = _load_one_member(path, pin)
        source.insert(0, _SOURCE_YEAR, np.int64(year))
        parts.append(source)
        provenance[str(year)] = record
    result = pd.concat(parts, ignore_index=True)
    result.attrs["provenance"] = provenance
    return result


def derive_cps_educational_attainment(
    person: pd.DataFrame,
    sources: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """``A_HGA`` and the class for every person, by exact Census identity.

    ``person`` is a US person table carrying ``source_year`` (the ASEC income
    year) and ``PERIDNUM``; ``sources`` is
    :func:`load_asec_attainment_sources`' result. Each person takes the member
    row of its own income year and ``PERIDNUM``, so a support clone takes its
    source person's value. The member's ``PH_SEQ`` must equal
    ``source_household_id`` (or, on an unpooled table without it, ``PH_SEQ``),
    and ``P_SEQ``, ``A_LINENO`` and ``A_AGE`` must agree wherever the table
    carries them.
    Refuses a person with no ``PERIDNUM`` (an ACS-spine row), an income year
    the sources do not cover, or a key the member lacks.
    """

    for column in (_SOURCE_YEAR, _JOIN_KEY):
        if column not in person.columns:
            _refuse(f"Person table lacks {column!r}; attainment joins on it.")
    if person[_JOIN_KEY].isna().any():
        _refuse(
            f"{int(person[_JOIN_KEY].isna().sum())} persons have no PERIDNUM. "
            "ACS-spine rows need the SCHL lane (attainment_from_schl), which "
            "the ACS-local build does not read yet."
        )
    years = _integer_codes(person[_SOURCE_YEAR], label="person source_year")
    try:
        keys = _exact_person_keys(person[_JOIN_KEY].reset_index(drop=True), "person")
    except ValueError as error:
        raise EducationalAttainmentError(str(error)) from error
    source_years = _integer_codes(sources[_SOURCE_YEAR], label="source source_year")
    uncovered = sorted(set(years.tolist()) - set(source_years.tolist()))
    if uncovered:
        _refuse(f"Attainment sources do not cover income year(s) {uncovered}.")

    a_hga = np.zeros(len(person), dtype=np.int64)
    identity_checked: dict[str, int] = {}
    joined: dict[str, int] = {}
    for year in sorted(set(years.tolist())):
        mask = years == year
        member = sources.loc[source_years == year].set_index(_JOIN_KEY)
        year_keys = keys[mask]
        missing = ~year_keys.isin(member.index)
        if missing.any():
            _refuse(
                f"{int(missing.sum())} income-year {year} persons have no row in "
                f"the Census member, e.g. PERIDNUM "
                f"{year_keys[missing].head(_EXAMPLES).tolist()}."
            )
        aligned = member.reindex(year_keys.to_numpy())
        for member_column in _IDENTITY_COLUMNS:
            frame_column = member_column
            if member_column == "PH_SEQ" and "source_household_id" in person.columns:
                # Pooling globalizes the frame's PH_SEQ; the raw Census PH_SEQ
                # survives as source_household_id (as the ED_VAL fill reads it).
                frame_column = "source_household_id"
            if frame_column not in person.columns:
                continue
            observed = pd.to_numeric(
                person.loc[mask, frame_column], errors="coerce"
            ).to_numpy(dtype=np.float64)
            expected = aligned[member_column].to_numpy(dtype=np.float64)
            disagree = ~np.isfinite(observed) | (observed != expected)
            if disagree.any():
                _refuse(
                    f"Person {frame_column} disagrees with the income-year {year} "
                    f"Census member {member_column} on {int(disagree.sum())} "
                    f"PERIDNUM-joined rows, e.g. PERIDNUM "
                    f"{year_keys[disagree].head(_EXAMPLES).tolist()}."
                )
            identity_checked[frame_column] = identity_checked.get(
                frame_column, 0
            ) + int(mask.sum())
        a_hga[mask] = aligned[US_ASEC_ATTAINMENT_SOURCE_COLUMN].to_numpy(dtype=np.int64)
        joined[str(year)] = int(mask.sum())

    classes = attainment_from_a_hga(a_hga)
    counts = pd.Series(classes).value_counts()
    return (
        a_hga,
        classes,
        {
            "operation": "exact_source_join",
            "join_key": "(source_year, PERIDNUM); many persons to one member row",
            "column": US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN,
            "column_role": "auxiliary",
            "source_column": US_ASEC_ATTAINMENT_SOURCE_COLUMN,
            "joined_persons_by_income_year": joined,
            "identity_columns_verified": identity_checked,
            "class_rows": {
                label: int(counts.get(label, 0))
                for label in US_EDUCATIONAL_ATTAINMENT_VALUES
            },
            "sources": sources.attrs.get("provenance", {}),
        },
    )


def with_us_cps_educational_attainment(
    frame: Frame,
    *,
    sources: pd.DataFrame,
) -> tuple[Frame, dict[str, object]]:
    """Append ``A_HGA`` and :data:`US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN`.

    A column the person table already carries is never overwritten: it must
    equal the derivation, which then leaves it as it is. Returns the frame
    and the derivation's provenance record.
    """

    if frame.schema != US_SCHEMA:
        _refuse("US educational attainment requires the US schema.")
    person = frame.table("person")
    a_hga, classes, record = derive_cps_educational_attainment(person, sources)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    out = tables["person"]
    for column, values in (
        (US_ASEC_ATTAINMENT_SOURCE_COLUMN, a_hga),
        (US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN, classes),
    ):
        if column in out.columns:
            carried = out[column].to_numpy()
            if column == US_ASEC_ATTAINMENT_SOURCE_COLUMN:
                carried = _integer_codes(carried, label=f"carried {column}")
            if not np.array_equal(carried.astype(values.dtype), values):
                _refuse(
                    f"The person table already carries {column} and it differs "
                    "from the Census derivation; it is never overwritten."
                )
            record.setdefault("columns_verified_equal", []).append(column)
            continue
        out[column] = values
        record.setdefault("columns_added", []).append(column)
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return result, record


def attainment_classes_from_metadata(metadata: Mapping[str, str]) -> frozenset[str]:
    """The class set a compiled attainment target counts."""

    raw = str(metadata.get(US_EDUCATIONAL_ATTAINMENT_CLASSES_METADATA_KEY, ""))
    classes = frozenset(part for part in raw.split(",") if part)
    unknown = sorted(classes - set(US_EDUCATIONAL_ATTAINMENT_CLASSES))
    if not classes or unknown:
        _refuse(
            f"Attainment target metadata names no class or unknown classes: {raw!r}."
        )
    return classes


def _as_bound(value: object) -> float:
    text = str(value)
    if text == "-inf":
        return -np.inf
    if text == "inf":
        return np.inf
    return float(text)


def attainment_target_person_indicator(
    attainment: object,
    age: object,
    metadata: Mapping[str, str],
) -> np.ndarray:
    """1.0 for each person a compiled attainment target counts, else 0.0.

    The target's age band is half-open, ``age_lower_bound <= age <
    age_upper_bound``, as every age materializer reads it; the state mask is
    the caller's, on the household collapse.
    """

    classes = attainment_classes_from_metadata(metadata)
    values = pd.Series(attainment).astype(object).to_numpy()
    unknown = ~pd.Series(values).isin(US_EDUCATIONAL_ATTAINMENT_VALUES).to_numpy()
    if unknown.any():
        _refuse(
            f"{US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN} has {int(unknown.sum())} values "
            "outside the reviewed classes."
        )
    ages = np.asarray(age, dtype=np.float64)
    lower = _as_bound(metadata.get("age_lower_bound", "-inf"))
    upper = _as_bound(metadata.get("age_upper_bound", "inf"))
    in_band = (ages >= lower) & (ages < upper)
    in_class = pd.Series(values).isin(sorted(classes)).to_numpy()
    return (in_band & in_class).astype(np.float64)


def us_cps_educational_attainment_summary(frame: Frame) -> dict[str, object]:
    """Weighted class shares overall and for the population 25 and over."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    classes = person[US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN].astype(object)
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=np.float64)
    adult = age >= 25
    total_adult = float(weights[adult].sum())
    summary: dict[str, object] = {
        "persons": len(person),
        "unknown_values": int((~classes.isin(US_EDUCATIONAL_ATTAINMENT_VALUES)).sum()),
        "missing_values": int(classes.isna().sum()),
        "weighted_population": float(weights.sum()),
        "weighted_population_25_plus": total_adult,
        "rows_by_class": {
            label: int(classes.eq(label).sum())
            for label in US_EDUCATIONAL_ATTAINMENT_VALUES
        },
        "share_25_plus_by_class": {
            label: (
                float(weights[adult & classes.eq(label).to_numpy()].sum()) / total_adult
                if total_adult > 0
                else 0.0
            )
            for label in US_EDUCATIONAL_ATTAINMENT_VALUES
        },
    }
    if "A_AGE" in person.columns:
        source_age = pd.to_numeric(person["A_AGE"], errors="coerce").to_numpy()
        niu = classes.eq(NOT_IN_UNIVERSE).to_numpy()
        child = source_age < US_EDUCATIONAL_ATTAINMENT_MINIMUM_AGE
        summary["universe_disagreements"] = int((niu != child).sum())
    shares = summary["share_25_plus_by_class"]
    summary["less_than_high_school_share_25_plus"] = float(
        shares[LESS_THAN_9TH_GRADE] + shares[GRADE_9_TO_12_NO_DIPLOMA]
    )
    summary["bachelors_or_higher_share_25_plus"] = float(
        shares[BACHELORS_DEGREE] + shares[GRADUATE_OR_PROFESSIONAL_DEGREE]
    )
    summary["less_than_high_school_share_band"] = list(
        _LESS_THAN_HIGH_SCHOOL_SHARE_BAND
    )
    summary["bachelors_or_higher_share_band"] = list(_BACHELORS_OR_HIGHER_SHARE_BAND)
    return summary


def us_cps_educational_attainment_signal_gate(frame: Frame) -> GateResult:
    """Require a complete, in-vocabulary, plausible attainment column."""

    person = frame.table("person")
    if US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN not in person.columns:
        return GateResult(
            name="cps_educational_attainment_signal",
            passed=False,
            failures=(
                f"person column {US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN} missing.",
            ),
            details={},
        )
    summary = us_cps_educational_attainment_summary(frame)
    failures: list[str] = []
    for key in ("unknown_values", "missing_values", "universe_disagreements"):
        count = int(summary.get(key, 0))
        if count:
            failures.append(f"{key}: {count}.")
    for share_key, band_key in (
        ("less_than_high_school_share_25_plus", "less_than_high_school_share_band"),
        ("bachelors_or_higher_share_25_plus", "bachelors_or_higher_share_band"),
    ):
        share = float(summary[share_key])
        low, high = summary[band_key]
        if not low <= share <= high:
            failures.append(f"{share_key} {share:.3f} outside [{low}, {high}].")
    return GateResult(
        name="cps_educational_attainment_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
