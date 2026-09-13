"""Pure US development-domain declarations over complete, supplied literal rows.

This module reads no source, constructs no Frame and applies no weights. Its
consistency checks cannot authenticate original membership, publisher weights,
source bytes or a receiving population. Every result requires those bindings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

PROTOCOL = "microcosm.us.survey-population-domains.v1"
DECLARATION = "experiments/us-survey-allocation-declaration-v1-20260907.md"
MAX_TOKEN_CHARS = 128
MAX_MEMBERS = 20
MAX_HOUSEHOLDS = 100_000
MAX_TOTAL_MEMBERS = 1_000_000


class DomainInputError(ValueError):
    """Static consistency refusal; never include raw identities or observations."""


class Source(Enum):
    ACS = "acs"
    ASEC = "asec"


class Qualifier(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class Status(Enum):
    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"
    REVIEW_REQUIRED = "review_required"


class Domain(Enum):
    SHARED_HOUSING = "shared_housing"
    RESIDUAL_HOUSING = "residual_housing"
    INSTITUTIONAL_GQ = "institutional_gq"
    NONINSTITUTIONAL_GQ = "noninstitutional_gq"


@dataclass(frozen=True, slots=True)
class HouseholdKey:
    source: Source
    source_year: int
    survey_year: int
    native_id: str


@dataclass(frozen=True, slots=True)
class AcsPerson:
    sporder: str
    age: str
    esr: str
    esr_state: str
    mil: str
    mil_state: str
    pwgtp: str
    household_key: HouseholdKey


@dataclass(frozen=True, slots=True)
class AsecPerson:
    peridnum: str
    a_lineno: str
    age: str
    prpertyp: str
    prpertyp_state: str
    household_key: HouseholdKey


@dataclass(frozen=True, slots=True)
class AcsHousehold:
    key: HouseholdKey
    typehugq: str
    np: str
    wgtp: str
    persons: tuple[AcsPerson, ...]


@dataclass(frozen=True, slots=True)
class AsecHousehold:
    key: HouseholdKey
    h_hhtype: str
    hrhtype: str
    h_livqrt: str
    h_numper: str
    hsup_wgt: str
    persons: tuple[AsecPerson, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    field: str
    literal: str
    state: str


@dataclass(frozen=True, slots=True)
class PersonDecision:
    original: AcsPerson | AsecPerson
    qualifier: Qualifier
    reason: str
    observations: tuple[Observation, ...]
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Annotation:
    source: str
    chosen_rule: str
    alternative: str
    status: str


@dataclass(frozen=True, slots=True)
class DomainShare:
    domain: Domain
    statistical_unit: str
    acs_share: Fraction
    asec_share: Fraction


@dataclass(frozen=True, slots=True)
class HouseholdDecision:
    original: AcsHousehold | AsecHousehold
    status: Status
    domain: Domain | None
    share: Fraction | None
    reason: str
    statistical_unit: str | None
    physical_housing_units: int | None
    publisher_weight_is_zero: bool | None
    people: tuple[PersonDecision, ...]
    observations: tuple[Observation, ...]
    annotations: tuple[Annotation, ...]

    @property
    def source_authenticated(self) -> bool:
        return False

    @property
    def population_binding_authenticated(self) -> bool:
        return False

    @property
    def release_eligible(self) -> bool:
        return False

    @property
    def required_bindings(self) -> tuple[str, ...]:
        return (
            "original_source_bytes",
            "complete_original_membership",
            "original_publisher_weight_authority",
            "source_period_and_residence_scope",
            "full_receiving_population_and_producer_edges",
            "actual_selection_probability",
        )

    @property
    def analysis_year(self) -> int:
        return 2024

    @property
    def income_reference(self) -> str:
        if self.original.key.source is Source.ASEC:
            return f"calendar_year_{self.original.key.source_year}"
        return "previous_12_months_at_each_2024_interview"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise DomainInputError(code)


def _token(value: str) -> str:
    _require(type(value) is str and len(value) <= MAX_TOKEN_CHARS, "LITERAL_TYPE_SIZE")
    return value


def _unsigned(raw: str, digits: int, maximum: int) -> int | None:
    _token(raw)
    if re.fullmatch(rf"[0-9]{{1,{digits}}}", raw) is None:
        return None
    value = int(raw)
    return value if value <= maximum else None


def _key(key: HouseholdKey) -> tuple[Source, int, int, str | int]:
    _require(type(key) is HouseholdKey and type(key.source) is Source, "HOUSEHOLD_KEY")
    _require(
        type(key.source_year) is int and type(key.survey_year) is int, "COHORT_TYPE"
    )
    _token(key.native_id)
    if key.source is Source.ACS:
        _require((key.source_year, key.survey_year) == (2024, 2024), "COHORT_IDENTITY")
        _require(
            re.fullmatch(r"2024(?:HU|GQ)[0-9]{7}", key.native_id) is not None
            and int(key.native_id[-7:]) > 0,
            "HOUSEHOLD_KEY",
        )
        native = key.native_id
    else:
        _require(
            key.source_year in (2022, 2023, 2024)
            and key.survey_year == key.source_year + 1,
            "COHORT_IDENTITY",
        )
        native = _unsigned(key.native_id, MAX_TOKEN_CHARS, 99999)
        _require(native is not None and native > 0, "HOUSEHOLD_KEY")
    return key.source, key.source_year, key.survey_year, native


def declaration() -> tuple[DomainShare, ...]:
    """The prospectively fixed coefficients, without applying them to weights."""
    return (
        DomainShare(
            Domain.SHARED_HOUSING,
            "occupied_housing_unit",
            Fraction(1, 2),
            Fraction(1, 2),
        ),
        DomainShare(
            Domain.RESIDUAL_HOUSING, "occupied_housing_unit", Fraction(1), Fraction(0)
        ),
        DomainShare(Domain.INSTITUTIONAL_GQ, "person", Fraction(1), Fraction(0)),
        DomainShare(Domain.NONINSTITUTIONAL_GQ, "person", Fraction(1), Fraction(0)),
    )


def reduce_qualifiers(values: tuple[Qualifier, ...]) -> Qualifier:
    """Existential reduction over known roster members, not membership proof."""
    _require(
        type(values) is tuple and 0 < len(values) <= MAX_MEMBERS, "QUALIFIER_ROSTER"
    )
    _require(all(type(v) is Qualifier for v in values), "QUALIFIER_TYPE")
    if Qualifier.TRUE in values:
        return Qualifier.TRUE
    if Qualifier.UNKNOWN in values:
        return Qualifier.UNKNOWN
    return Qualifier.FALSE


def _acs_state(age: str, raw: str, minimum: int, codes: tuple[str, ...]) -> str:
    # Exact vocabulary of acs_person_coverage_columns._field_state. No owner import.
    value = _unsigned(age, 2, 99)
    if value is None:
        return "age_unresolved"
    if value < minimum:
        return "outside_age_universe" if raw == "" else "value_below_age_universe"
    if raw == "":
        return "missing_in_universe"
    return "observed_code" if raw in codes else "unlabelled_code"


def _asec_state(raw: str) -> str:
    # Exact vocabulary of asec_person_coverage_source._state.
    if raw in ("1", "2", "3"):
        return "observed_code"
    if raw == "":
        return "blank_unresolved"
    if raw in ("-4", "-3", "-2", "-1", "0"):
        return "unlabelled_in_range"
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", raw) and raw != "-0":
        return "out_of_range"
    return "malformed_token"


def _person(person: AcsPerson | AsecPerson) -> PersonDecision:
    is_acs = type(person) is AcsPerson
    age = _unsigned(person.age, 2 if is_acs else 19, 99 if is_acs else 2**63 - 1)
    observations = [
        Observation(
            "AGEP" if is_acs else "A_AGE",
            person.age,
            "observed_integer" if age is not None else "age_unresolved",
        )
    ]
    diagnostics = []
    if is_acs:
        _token(person.pwgtp)
        for raw, state in (
            (person.esr, person.esr_state),
            (person.mil, person.mil_state),
        ):
            _token(raw)
            _token(state)
        _require(
            person.esr_state
            == _acs_state(person.age, person.esr, 16, ("1", "2", "3", "4", "5", "6")),
            "FIELD_STATE_MISMATCH",
        )
        _require(
            person.mil_state
            == _acs_state(person.age, person.mil, 17, ("1", "2", "3", "4")),
            "FIELD_STATE_MISMATCH",
        )
        observations.extend(
            (
                Observation("ESR", person.esr, person.esr_state),
                Observation("MIL", person.mil, person.mil_state),
            )
        )
        true = person.esr in ("1", "2", "3", "6")
        false = person.esr in ("4", "5")
        if (
            person.mil_state == "observed_code"
            and person.esr_state == "observed_code"
            and (
                (person.mil == "1" and true)
                or (person.mil in ("2", "3", "4") and false)
            )
        ):
            diagnostics.append("mil_esr_disagreement")
        if person.mil_state != "observed_code":
            diagnostics.append("mil_" + person.mil_state)
        contradiction = False
    else:
        _token(person.prpertyp)
        _token(person.prpertyp_state)
        _require(
            person.prpertyp_state == _asec_state(person.prpertyp),
            "FIELD_STATE_MISMATCH",
        )
        observations.append(
            Observation("PRPERTYP", person.prpertyp, person.prpertyp_state)
        )
        true, false = person.prpertyp == "2", person.prpertyp == "3"
        contradiction = age is not None and age >= 16 and person.prpertyp == "1"
    if age is None:
        qualifier, reason = Qualifier.UNKNOWN, "source_age_unresolved"
    elif age < 16:
        qualifier, reason = Qualifier.FALSE, "source_age_below_16"
    elif contradiction:
        qualifier, reason = Qualifier.UNKNOWN, "adult_age_with_child_type"
    elif true:
        qualifier, reason = Qualifier.TRUE, "source_civilian_proxy"
    elif false:
        qualifier, reason = Qualifier.FALSE, "source_armed_forces_proxy"
    else:
        qualifier, reason = Qualifier.UNKNOWN, "required_person_status_unresolved"
    return PersonDecision(
        person, qualifier, reason, tuple(observations), tuple(diagnostics)
    )


def _members(row: AcsHousehold | AsecHousehold) -> tuple[PersonDecision, ...]:
    _require(
        type(row.persons) is tuple and len(row.persons) <= MAX_MEMBERS,
        "MEMBER_TUPLE_BOUND",
    )
    household = _key(row.key)
    seen_keys, seen_lines = set(), set()
    expected = AcsPerson if type(row) is AcsHousehold else AsecPerson
    for person in row.persons:
        _require(type(person) is expected, "MEMBER_TYPE")
        _require(_key(person.household_key) == household, "MEMBER_HOUSEHOLD_KEY")
        raw_line = person.sporder if expected is AcsPerson else person.a_lineno
        line = _unsigned(
            raw_line,
            2 if expected is AcsPerson else 19,
            20 if expected is AcsPerson else 2**63 - 1,
        )
        _require(
            line is not None and line > 0 and line not in seen_lines, "MEMBER_LINE_KEY"
        )
        seen_lines.add(line)
        person_key = line if expected is AcsPerson else _token(person.peridnum)
        if expected is AsecPerson:
            _require(
                re.fullmatch(r"[0-9]{22}", person_key) is not None, "MEMBER_PERSON_KEY"
            )
        _require(person_key not in seen_keys, "MEMBER_DUPLICATE_KEY")
        seen_keys.add(person_key)
    return tuple(_person(person) for person in row.persons)


def _field(
    name: str, raw: str, digits: int, maximum: int, minimum: int = 0
) -> tuple[int | None, Observation]:
    value = _unsigned(raw, digits, maximum)
    known = value is not None and value >= minimum
    state = (
        "valid"
        if known
        else "missing"
        if raw == ""
        else "malformed"
        if re.fullmatch(rf"[0-9]{{1,{digits}}}", raw) is None
        else "unlabelled"
    )
    return value if known else None, Observation(name, raw, state)


def classify_household(row: AcsHousehold | AsecHousehold) -> HouseholdDecision:
    """Classify supplied complete rows; unresolved results authorize no allocation.

    Key/tuple/count inconsistencies raise. Unknown required observations retain
    their literals and produce REVIEW_REQUIRED, never residual by default.
    Source and population authority cannot be supplied as a boolean argument.
    """
    _require(type(row) in (AcsHousehold, AsecHousehold), "HOUSEHOLD_TYPE")
    key = _key(row.key)
    is_acs = type(row) is AcsHousehold
    _require(key[0] is (Source.ACS if is_acs else Source.ASEC), "HOUSEHOLD_SOURCE")
    people = _members(row)
    observations = []
    zero = None

    def field(name, raw, digits, maximum, minimum=0):
        value, observation = _field(name, raw, digits, maximum, minimum)
        observations.append(observation)
        return value

    def result(status, reason, domain=None, share=None, unit=None, housing_units=None):
        return HouseholdDecision(
            row,
            status,
            domain,
            share,
            reason,
            unit,
            housing_units,
            zero,
            people,
            tuple(observations),
            (
                Annotation(
                    DECLARATION,
                    "source_age_16_ESR_only_or_PRPERTYP",
                    "joint_MIL_ESR_or_other_age_threshold",
                    "prespecified_development_choice",
                ),
                Annotation(
                    DECLARATION,
                    "fixed_domain_shares_and_structural_exclusions",
                    "estimated_overlap_or_optimized_shares",
                    "not_empirically_validated",
                ),
                Annotation(
                    DECLARATION,
                    reason,
                    "no_silent_residual_or_donor_admission",
                    status.value,
                ),
            ),
        )

    count = field(
        "NP" if is_acs else "H_NUMPER",
        row.np if is_acs else row.h_numper,
        2,
        20 if is_acs else 16,
    )
    if count is not None:
        _require(count == len(row.persons), "REPORTED_MEMBERSHIP_COUNT")
    if is_acs:
        housing = field("TYPEHUGQ", row.typehugq, 1, 3, 1)
        weight = field("WGTP", row.wgtp, 4, 9999)
        if count is None or housing is None:
            return result(Status.REVIEW_REQUIRED, "acs_household_condition_unresolved")
        if row.key.native_id[4:6] != ("HU" if housing == 1 else "GQ"):
            return result(Status.REVIEW_REQUIRED, "acs_serialno_housing_conflict")
        if housing == 1 and count == 0:
            return result(
                Status.EXCLUDED, "acs_vacancy", share=Fraction(0), housing_units=0
            )
        if housing in (2, 3):
            if count != 1 or weight != 0:
                return result(Status.REVIEW_REQUIRED, "acs_gq_placeholder_conflict")
            person_weight = field("PWGTP", row.persons[0].pwgtp, 4, 9999, 1)
            if person_weight is None:
                return result(Status.REVIEW_REQUIRED, "publisher_weight_unresolved")
            zero = False
            return result(
                Status.ELIGIBLE,
                "acs_gq_allocation",
                Domain.INSTITUTIONAL_GQ if housing == 2 else Domain.NONINSTITUTIONAL_GQ,
                Fraction(1),
                "person",
                0,
            )
        if weight is None or weight <= 0:
            return result(Status.REVIEW_REQUIRED, "publisher_weight_unresolved")
        zero = False
    else:
        interview = field("H_HHTYPE", row.h_hhtype, 1, 3, 1)
        household_type = field("HRHTYPE", row.hrhtype, 2, 10)
        living = field("H_LIVQRT", row.h_livqrt, 2, 12, 1)
        weight = field("HSUP_WGT", row.hsup_wgt, MAX_TOKEN_CHARS, 999999999)
        if count is None:
            return result(Status.REVIEW_REQUIRED, "asec_membership_count_unresolved")
        if row.key.source_year != 2024:
            return result(Status.EXCLUDED, "asec_older_cohort", share=Fraction(0))
        if interview in (2, 3):
            return result(Status.EXCLUDED, "asec_noninterview", share=Fraction(0))
        if interview is None or living is None or household_type is None:
            return result(Status.REVIEW_REQUIRED, "asec_household_condition_unresolved")
        if living == 11:
            return result(
                Status.REVIEW_REQUIRED,
                "asec_student_quarters_source_review",
                share=Fraction(0),
                housing_units=0,
            )
        if living >= 8:
            return result(
                Status.EXCLUDED, "asec_nonhousing", share=Fraction(0), housing_units=0
            )
        if household_type not in range(1, 9) or count == 0:
            return result(Status.REVIEW_REQUIRED, "asec_housing_membership_conflict")
        if weight is None:
            return result(Status.REVIEW_REQUIRED, "publisher_weight_unresolved")
        zero = weight == 0
    predicate = reduce_qualifiers(tuple(person.qualifier for person in people))
    if predicate is Qualifier.UNKNOWN:
        return result(Status.REVIEW_REQUIRED, "household_qualifier_unresolved")
    if predicate is Qualifier.TRUE:
        return result(
            Status.ELIGIBLE,
            "chosen_shared_housing",
            Domain.SHARED_HOUSING,
            Fraction(1, 2),
            "occupied_housing_unit",
            1,
        )
    return result(
        Status.ELIGIBLE if is_acs else Status.EXCLUDED,
        "acs_known_residual_housing" if is_acs else "asec_residual_zero_allocation",
        Domain.RESIDUAL_HOUSING,
        Fraction(1) if is_acs else Fraction(0),
        "occupied_housing_unit",
        1,
    )


def classify_households(
    rows: tuple[AcsHousehold | AsecHousehold, ...],
) -> tuple[HouseholdDecision, ...]:
    """Classify a bounded batch, rejecting household and cohort-person collisions."""
    _require(type(rows) is tuple and 0 < len(rows) <= MAX_HOUSEHOLDS, "HOUSEHOLD_BATCH")
    keys, members = set(), 0
    for row in rows:
        _require(type(row) in (AcsHousehold, AsecHousehold), "HOUSEHOLD_TYPE")
        key = _key(row.key)
        _require(key not in keys, "DUPLICATE_HOUSEHOLD_KEY")
        keys.add(key)
        _require(
            type(row.persons) is tuple and len(row.persons) <= MAX_MEMBERS,
            "MEMBER_TUPLE_BOUND",
        )
        members += len(row.persons)
        _require(members <= MAX_TOTAL_MEMBERS, "BATCH_MEMBER_BOUND")
    person_keys = set()
    for row in rows:
        if type(row) is AsecHousehold:
            for person in row.persons:
                _require(type(person) is AsecPerson, "MEMBER_TYPE")
                literal = _token(person.peridnum)
                _require(
                    re.fullmatch(r"[0-9]{22}", literal) is not None,
                    "MEMBER_PERSON_KEY",
                )
                person_key = (row.key.source_year, literal)
                _require(person_key not in person_keys, "DUPLICATE_COHORT_PERSON_KEY")
                person_keys.add(person_key)
    return tuple(classify_household(row) for row in rows)
