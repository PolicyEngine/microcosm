"""Prospective December SIPP housing predictor observations, measured only.

Additive to :mod:`microcosm.build.sipp_housing_measurements`, whose admission
ladder this module cannot reach. The predictor pass runs *after*
``december_housing`` has already classified every household, receives that
classification as an input, and returns a parallel evidence tuple. Nothing here
is an argument to ``_exclusion``: the old measurement ladder is unchanged not
because this module promises it, but because the call graph gives it no way to
participate. ``december_predictors`` regroups the same supplied rows and refuses
``PREDICTOR_ROSTER_DISAGREEMENT`` rather than adopting a second household
identity.

What this module does **not** do: annualize a monthly amount, rebase a price,
bridge 2022 to any other year, log or floor an income, fit, draw, select a donor
mask, bind a model input, map a recipient, or invent a sub-state geography. The
nine columns are transcribed from the pinned 2023 SIPP data dictionary and read
as they are published; where a field's actual semantics do not support the use
another document proposed for it, the observation is preserved and the
limitation is declared instead.

See docs/us-sipp-december-housing-projection.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from . import sipp_housing_measurements as hm
from . import sipp_measurements as m

PREDICTOR_RULE_ID = "sipp_december_housing_predictors_v1"
PersonKey = m.PersonKey

# 18 is the published boundary of RHNUMU18's own concept ("under 18 years"), so
# the roster counterpart derived from TAGE_EHC uses the same boundary rather
# than a policy age chosen here.
CHILD_AGE_BOUNDARY = 18

# The one column shared with the frozen sixteen-field producer, and the reading
# both must agree on. ``test_us_sipp_housing_projection.py`` asserts the two.
SHARED_WITH_FROZEN_PRODUCER = "TAGE_EHC"
FROZEN_PRODUCER_AGE_READING = {"low": 0, "high": 90, "sentinel": -999}


class HousingPredictorError(m.MeasurementError):
    """A requested predictor observation cannot be supplied under this contract."""


# TEHC_ST's published value list, transcribed from the pinned dictionary. It is
# 53 codes, not 51: 60 and 61 are published area codes that are not a state or
# the District of Columbia, and they are never folded into a state factor here.
_STATE_CODES = (
    1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 60, 61,
)  # fmt: skip
_STATE_LABELS = (
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
    "Puerto Rico and Island Areas",
    "Foreign Country",
)
NON_STATE_AREA_CODES = (60, 61)

_ALL_PERSONS_UNIVERSE = "THHLDSTATUS in (1,2,3,4)"

_PREDICTOR_CATEGORIES: Mapping[str, hm._Category] = {
    "TEHC_ST": hm._Category(
        "AEHC_ST",
        _STATE_CODES,
        _STATE_LABELS,
        "person_month",
        "monthly_state_of_residence",
        _ALL_PERSONS_UNIVERSE,
        "All persons.",
        3310,
    ),
}

_PREDICTOR_AMOUNTS: Mapping[str, hm._Amount] = {
    "RHNUMPER": hm._Amount(
        "AHNUMPER",
        1,
        20,
        "household_concept_person_month",
        "number_of_persons_in_household_this_month",
        "December_2022_household_member_count",
        _ALL_PERSONS_UNIVERSE,
        "All persons present at this month",
        890,
        unit="count",
        sentinel=0,
        kind="count",
    ),
    "RHNUMU18": hm._Amount(
        "AHNUMU18",
        0,
        20,
        "household_concept_person_month",
        "number_of_persons_in_household_under_18_years_this_month",
        "December_2022_household_member_count_under_18",
        _ALL_PERSONS_UNIVERSE,
        "All persons present at this month",
        891,
        unit="count",
        sentinel=-1,
        kind="count",
    ),
    "THTOTINC": hm._Amount(
        "AHTOTINC",
        -10_000_000_000,
        10_000_000_000,
        "household_concept_person_month",
        "sum_of_monthly_earnings_and_income_received_by_household_members_age_15"
        "_and_older_as_well_as_SSI_payments_received_by_children_under_age_15",
        "December_2022_household_monthly_total_income",
        _ALL_PERSONS_UNIVERSE,
        "All people",
        2865,
        unit="USD_nominal_2022_monthly",
        sentinel=-10_000_000_001,
        kind="amount",
    ),
    "TAGE_EHC": hm._Amount(
        None,
        0,
        90,
        "person_month",
        "monthly_age_during_the_reference_period",
        "December_2022_person_monthly_age",
        _ALL_PERSONS_UNIVERSE,
        "All persons present at this month",
        796,
        unit="years",
        # TAGE_EHC is the one column this projection shares with the frozen
        # sixteen-field producer, which reads it as ``_number(..., 0, 90, -999)``
        # to gate its retirement-income and bank-asset universes. The domain and
        # the out-of-domain token are kept identical here so the two readings of
        # the same column cannot diverge; this lane introduces no second
        # contract for it.
        sentinel=-999,
        kind="age",
        unflagged_origin="published_source_value_without_status_flag",
    ),
}

PREDICTOR_CATEGORIES = tuple(_PREDICTOR_CATEGORIES)
PREDICTOR_AMOUNTS = tuple(_PREDICTOR_AMOUNTS)
PREDICTOR_VALUE_FIELDS = (*PREDICTOR_CATEGORIES, *PREDICTOR_AMOUNTS)

# Ordered lexical columns this pass requires, value field followed by its
# published status flag. TAGE_EHC prints "Status Flag: NA" and has none.
PREDICTOR_FIELDS = (
    "TEHC_ST",
    "AEHC_ST",
    "RHNUMPER",
    "AHNUMPER",
    "RHNUMU18",
    "AHNUMU18",
    "THTOTINC",
    "AHTOTINC",
    "TAGE_EHC",
)

# Fields resolved to one household value by the frozen kernel's copy-agreement
# rule, and fields kept at the person grain. ``TEHC_ST`` is in the first group
# because a December residence has one location, **not** because the dictionary
# says the column is constant within a household: it does not. Agreement is
# therefore checked, and a disagreement is a copy conflict rather than a vote.
HOUSEHOLD_RESOLVED_FIELDS = ("TEHC_ST", "RHNUMPER", "RHNUMU18", "THTOTINC")
PERSON_RESOLVED_FIELDS = ("TAGE_EHC",)

PREDICTOR_PURPOSE = {
    "TEHC_ST": "prospective_state_or_area_of_residence_conditioning_variable",
    "AEHC_ST": "transcribed_allocation_flag_for_TEHC_ST",
    "RHNUMPER": "prospective_publisher_household_size_conditioning_variable",
    "AHNUMPER": "transcribed_allocation_flag_for_RHNUMPER",
    "RHNUMU18": "prospective_children_under_18_conditioning_variable",
    "AHNUMU18": "transcribed_allocation_flag_for_RHNUMU18",
    "THTOTINC": "prospective_household_monthly_income_conditioning_variable",
    "AHTOTINC": "transcribed_allocation_flag_for_THTOTINC",
    "TAGE_EHC": "prospective_member_age_conditioning_variable_and_child_roster_basis",
}

# Dictionary "Length:" line, read off each entry's own page.
PREDICTOR_LENGTHS = {
    "TEHC_ST": 2,
    "RHNUMPER": 2,
    "RHNUMU18": 2,
    "THTOTINC": 12,
    "TAGE_EHC": 2,
}

# How each field reaches a household value, declared rather than assumed.
PREDICTOR_RESOLUTION = {
    "TEHC_ST": "household_copy_agreement_checked_not_assumed",
    "RHNUMPER": "household_copy_agreement",
    "RHNUMU18": "household_copy_agreement",
    "THTOTINC": "household_copy_agreement",
    "TAGE_EHC": "person_observation_maximum_over_admissible_december_members",
}

# What each field does *not* support, read from its own dictionary entry. These
# are the conflicts between a published semantic and a use another document
# proposed; the observation is preserved and the limitation is declared.
PREDICTOR_LIMITATIONS = {
    "TEHC_ST": (
        "The published value list is 53 codes, not 51 and not the range 01-56: "
        "60 'Puerto Rico and Island Areas' and 61 'Foreign Country' are "
        "published codes that are not a state or the District of Columbia. A "
        "51-level state factor is not this column's domain.",
        "The dictionary prints File: Person-month and does not declare the "
        "column constant within a household, so a household value exists only "
        "when every December member's observation agrees.",
        "No sub-state geography is published on this column and none is derived here.",
    ),
    "RHNUMPER": (
        "The dictionary's own paired variant RHNUMPERWT2 (printed page 899) is "
        "described 'Number of persons in household this month (with Type 2 "
        "persons)'. RHNUMPER's description does not carry that phrase. The "
        "contrast is the pair's; no statement on RHNUMPER's own page says what "
        "it excludes, and the roster's treatment of Type 2 persons is not "
        "adjudicated here. Both counts are retained.",
    ),
    "RHNUMU18": (
        "RHNUMU18WT2 (printed page 900) is the paired Type-2-inclusive variant, "
        "under the same reading as RHNUMPER.",
        "The roster counterpart derived from TAGE_EHC is established only when "
        "every December member carries an admissible age observation.",
    ),
    "THTOTINC": (
        "The published amount is a December 2022 *monthly* sum. No "
        "annualization, price rebasing, flooring or logarithm is performed "
        "here; each would be a separately declared assumption.",
        "The published composition is the dictionary's own: earnings and income "
        "of household members age 15 and older plus SSI payments received by "
        "children under 15. It is not a calendar-year income and not an ASEC "
        "HTOTVAL equivalent.",
        "The published minimum is negative, so a negative value is an in-domain "
        "observation and is preserved signed.",
        "The period-consistent alternative THCYINCPOV (printed page 2880, "
        "status flag AHCYINCPOV) is not carried by this projection.",
    ),
    "TAGE_EHC": (
        "This is the only column the projection shares with the frozen "
        "sixteen-field producer. Its domain 0-90 and out-of-domain token -999 "
        "are read exactly as that producer reads them, so the shared column has "
        "one contract and not two.",
        "The dictionary prints 'Status Flag: NA', so no allocation quality "
        "exists for this column: reported and imputed ages cannot be "
        "distinguished, unlike every flagged field in this projection.",
        "The published maximum is 90. A household maximum of 90 is recorded as "
        "sitting at the published domain maximum; the dictionary does not call "
        "it a topcode on this page and none is inferred.",
        "A household maximum derived while some member observations are "
        "inadmissible is a lower bound, not the maximum.",
    ),
}

# Columns a reader might expect beside these and the reason each is absent, so
# the width of the projection stays a decision rather than an omission.
PREDICTOR_FIELDS_DELIBERATELY_ABSENT = {
    "TEHC_METRO": "metro_status_unused_in_this_increment",
    "AEHC_METRO": "metro_status_unused_in_this_increment",
    "RHNUMPERWT2": "type_2_inclusive_variant_not_carried",
    "RHNUMU18WT2": "type_2_inclusive_variant_not_carried",
    "THTOTINCT2": "type_2_inclusive_variant_not_carried",
    "THCYINCPOV": "period_consistent_income_alternative_not_carried",
    "AHCYINCPOV": "period_consistent_income_alternative_not_carried",
    "THHLDSTATUS": "published_universe_gate_of_all_nine_columns_not_carried",
    "ERELRPE": "reference_person_relationship_not_carried",
    "ARELRP": "reference_person_relationship_not_carried",
}

# Exclusive prospective-validity ladder. It is reported beside the measurement
# admission ladder and is not part of it: no state below ever changes a
# household's ``exclusion``.
PREDICTOR_STATUS_ORDER = (
    "predictor_copy_conflict",
    "predictor_suppressed",
    "predictor_not_in_universe",
    "predictor_unknown",
    "predictor_age_coverage_incomplete",
)
AGE_COVERAGE_STATES = ("complete", "partial", "none")
DISAGREEMENT_STATES = (
    "agree",
    "disagree",
    "publisher_conflict",
    "publisher_unknown",
    "roster_coverage_incomplete",
)
INCOME_SIGN_STATES = ("negative", "zero", "positive", "unknown")


@dataclass(frozen=True)
class HouseholdPredictors:
    """One December residence's predictor evidence; no fit admission is implied."""

    key: tuple[str, int]
    members: tuple[PersonKey, ...]
    household_measurements: tuple[m.Measurement, ...]
    person_age_measurements: tuple[m.Measurement, ...]
    maximum_member_age: m.Measurement
    status: str
    diagnostics: Mapping[str, object]

    def household(self, name: str) -> m.Measurement:
        for measure in self.household_measurements:
            if measure.concept == name:
                return measure
        raise HousingPredictorError("UNKNOWN_PREDICTOR_FIELD:" + name)


def required_fields() -> tuple[str, ...]:
    """Every lexical column the predictor pass reads, identity columns included."""
    return (*PREDICTOR_FIELDS, "ERESIDENCEID")


def _observe(key: PersonKey, month: int, row: m.SippRow, name: str) -> m.Observation:
    if name in _PREDICTOR_CATEGORIES:
        return hm._observe_category(key, month, row, name, _PREDICTOR_CATEGORIES[name])
    return hm._observe_amount(key, month, row, name, _PREDICTOR_AMOUNTS[name])


def _spec(name: str) -> hm._Category | hm._Amount:
    if name in _PREDICTOR_CATEGORIES:
        return _PREDICTOR_CATEGORIES[name]
    return _PREDICTOR_AMOUNTS[name]


def _basis(name: str) -> str:
    spec = _spec(name)
    if isinstance(spec, hm._Category):
        return "December_2022_person_month_category"
    return spec.basis


def _unit(name: str) -> str:
    spec = _spec(name)
    return "code" if isinstance(spec, hm._Category) else spec.unit


def _age_admissible(measure: m.Measurement) -> bool:
    """One member's age observation is usable as published, with no reason against it."""
    return measure.value is not None and not measure.reasons


def _maximum_member_age(
    source: m.SippSource, ages: Sequence[m.Measurement]
) -> tuple[m.Measurement, str, int]:
    """Maximum over admissible member ages only, with its coverage stated.

    A partial coverage produces a **lower bound**, never a maximum: an
    inadmissible member could be older than every member that was read. The
    measurement still carries the derived value, because discarding a real
    observation would lose evidence, and the reason and the coverage state say
    what it is.
    """
    spec = _PREDICTOR_AMOUNTS["TAGE_EHC"]
    admissible = [a for a in ages if _age_admissible(a)]
    covered = len(admissible)
    if not ages:
        coverage = "none"
    elif covered == 0:
        coverage = "none"
    elif covered == len(ages):
        coverage = "complete"
    else:
        coverage = "partial"
    reasons: list[str] = []
    if coverage != "complete":
        reasons.append("age_coverage:" + coverage)
    value = max((a.value for a in admissible), default=None)
    if value is None:
        reasons.append("TAGE_EHC:no_admissible_member_observation")
    observations = tuple(o for a in ages for o in a.observations)
    return (
        hm._build(
            "sipp_december_household_maximum_member_age",
            "TAGE_EHC_maximum_over_admissible_december_members",
            source,
            "December_2022_household_maximum_member_age",
            spec.unit,
            observations,
            value,
            "household_maximum_member_age"
            if value is not None and coverage == "complete"
            else "household_maximum_member_age_lower_bound"
            if value is not None
            else "unknown_household_maximum_member_age",
            reasons,
            rule_id=PREDICTOR_RULE_ID,
        ),
        coverage,
        covered,
    )


def _status(
    household: Mapping[str, m.Measurement],
    maximum_age: m.Measurement,
    coverage: str,
) -> str:
    """Exclusive prospective-validity state; never an admission decision."""
    classes = {name: hm.measurement_class(household[name]) for name in household}
    if any(state == "copy_conflict" for state in classes.values()):
        return "predictor_copy_conflict"
    # Withheld and out-of-universe evidence are each kept apart from having no
    # answer to give, on the same rule the assistance classification follows.
    if any(state == "suppressed" for state in classes.values()):
        return "predictor_suppressed"
    if any(state == "not_in_universe" for state in classes.values()):
        return "predictor_not_in_universe"
    if any(household[name].value is None for name in household):
        return "predictor_unknown"
    if maximum_age.value is None or coverage != "complete":
        return "predictor_age_coverage_incomplete"
    return "predictor_complete"


def _income_sign(measure: m.Measurement) -> str:
    if measure.value is None:
        return "unknown"
    if measure.value < 0:
        return "negative"
    return "zero" if measure.value == 0 else "positive"


def _agreement(publisher: m.Measurement, roster: int | None) -> str:
    """Compare a publisher count with its roster counterpart, correcting neither.

    A roster counterpart that could not be established is reported as such
    rather than as agreement, and a publisher measurement that did not resolve
    is separated into a copy conflict and every other non-resolving state.
    """
    if roster is None:
        return "roster_coverage_incomplete"
    if "household_copy_conflict" in publisher.reasons:
        return "publisher_conflict"
    if publisher.value is None:
        return "publisher_unknown"
    return "agree" if int(publisher.value) == roster else "disagree"


def _household_predictors(
    source: m.SippSource,
    key: tuple[str, int],
    members: Sequence[tuple[PersonKey, m.SippRow]],
) -> HouseholdPredictors:
    household_measures = []
    for name in HOUSEHOLD_RESOLVED_FIELDS:
        observations = tuple(_observe(person, 12, row, name) for person, row in members)
        household_measures.append(
            hm._copied(
                "sipp_december_" + name.lower(),
                name,
                source,
                _basis(name),
                _unit(name),
                observations,
                rule_id=PREDICTOR_RULE_ID,
            )
        )
    ages = []
    for person, row in members:
        observation = _observe(person, 12, row, "TAGE_EHC")
        ages.append(
            hm._build(
                "sipp_december_tage_ehc",
                "TAGE_EHC",
                source,
                _basis("TAGE_EHC"),
                _unit("TAGE_EHC"),
                (observation,),
                observation.value if not observation.reasons else None,
                "published_person_month_age"
                if observation.usable
                else "unknown_person_month_age",
                observation.reasons,
                rule_id=PREDICTOR_RULE_ID,
            )
        )
    maximum_age, coverage, covered = _maximum_member_age(source, ages)
    lookup = {v.concept: v for v in household_measures}
    status = _status(lookup, maximum_age, coverage)
    roster_size = len(members)
    roster_children = (
        sum(1 for a in ages if _age_admissible(a) and a.value < CHILD_AGE_BOUNDARY)
        if coverage == "complete"
        else None
    )
    state = lookup["TEHC_ST"]
    state_code = int(state.value) if state.value is not None else None
    diagnostics = {
        "predictor_status": status,
        "predictor_field_classes": {
            name: hm.measurement_class(lookup[name])
            for name in HOUSEHOLD_RESOLVED_FIELDS
        },
        "predictor_field_quality": {
            name: hm.measurement_quality(lookup[name])
            for name in HOUSEHOLD_RESOLVED_FIELDS
        },
        # Both counts are retained. Neither is corrected against the other, and
        # the projection carries no THHLDSTATUS, so the Type 2 treatment behind
        # a disagreement is recorded as unadjudicated rather than explained.
        "publisher_household_size_known": lookup["RHNUMPER"].value is not None,
        "roster_december_member_count": roster_size,
        "household_size_agreement": _agreement(lookup["RHNUMPER"], roster_size),
        "household_size_difference": (
            int(lookup["RHNUMPER"].value) - roster_size
            if lookup["RHNUMPER"].value is not None
            else None
        ),
        "roster_december_members_under_18": roster_children,
        "children_under_18_agreement": _agreement(lookup["RHNUMU18"], roster_children),
        "age_coverage": coverage,
        "age_covered_members": covered,
        "maximum_age_is_lower_bound": maximum_age.value is not None
        and coverage != "complete",
        "maximum_age_at_domain_maximum": maximum_age.value is not None
        and maximum_age.value == _PREDICTOR_AMOUNTS["TAGE_EHC"].high,
        "income_sign": _income_sign(lookup["THTOTINC"]),
        "income_at_domain_minimum": lookup["THTOTINC"].value is not None
        and lookup["THTOTINC"].value == _PREDICTOR_AMOUNTS["THTOTINC"].low,
        "income_at_domain_maximum": lookup["THTOTINC"].value is not None
        and lookup["THTOTINC"].value == _PREDICTOR_AMOUNTS["THTOTINC"].high,
        "state_is_published_non_state_area": state_code in NON_STATE_AREA_CODES,
        # EEHC_RENTSUB's published universe carries the arm ``TEHC_ST <= '60'``,
        # which version 2 could not see because TEHC_ST was not carried. It is
        # observed here and **changes no classification**: the assistance class,
        # the person-month measurements and the exclusion ladder are untouched.
        "eehc_rentsub_state_arm": "state_unknown"
        if state_code is None
        else "state_code_at_most_60"
        if state_code <= 60
        else "state_code_above_60",
    }
    return HouseholdPredictors(
        key,
        tuple(person for person, _ in members),
        tuple(household_measures),
        tuple(ages),
        maximum_age,
        status,
        diagnostics,
    )


def december_predictors(
    rows: Sequence[m.SippRow],
    households: Sequence[hm.HousingHousehold],
    *,
    source: m.SippSource,
    complete_membership: bool,
    target_year: int = 2022,
) -> tuple[HouseholdPredictors, ...]:
    """Observe the predictor columns on the already-classified housing roster.

    ``households`` is an input, never an output: this pass reads the roster the
    housing measurement already fixed and refuses if its own regrouping of the
    same rows disagrees. No value produced here reaches ``_exclusion``.
    """
    m._check_source(source, target_year)
    if type(complete_membership) is not bool:
        raise HousingPredictorError("INVALID_MEMBERSHIP_ASSERTION")
    prepared = m._prepare(rows, required_fields())
    groups: dict[tuple[str, int], list[tuple[PersonKey, m.SippRow]]] = {}
    for key, month, row in prepared:
        if month != 12:
            continue
        _, residence = m._number(row.values, "ERESIDENCEID", 100001, 400999, -999)
        if residence is None:
            raise HousingPredictorError("INVALID_DECEMBER_RESIDENCE")
        groups.setdefault((key[0], int(residence)), []).append((key, row))
    regrouped = tuple(
        (key, tuple(person for person, _ in members))
        for key, members in sorted(groups.items())
    )
    if regrouped != tuple((h.key, h.members) for h in households):
        raise HousingPredictorError("PREDICTOR_ROSTER_DISAGREEMENT")
    return tuple(
        _household_predictors(source, household.key, groups[household.key])
        for household in households
    )


def _counter(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def predictor_diagnostics(
    predictors: Sequence[HouseholdPredictors],
    households: Sequence[hm.HousingHousehold],
) -> dict:
    """Aggregate prospective predictor validity, reported beside the ladder.

    Every count is unweighted and aggregate. ``households`` supplies only each
    household's already-decided ``exclusion``, so the admitted subset can be
    reported separately; nothing here feeds back into that decision. As with the
    version 2 diagnostics, a class holding a single household discloses that
    household's value for the class, which now includes its state code.
    """
    if len(predictors) != len(households) or any(
        p.key != h.key for p, h in zip(predictors, households, strict=True)
    ):
        raise HousingPredictorError("PREDICTOR_HOUSEHOLD_ALIGNMENT")
    admitted = tuple(
        p for p, h in zip(predictors, households, strict=True) if h.exclusion is None
    )

    def _state_label(predictor: HouseholdPredictors) -> str:
        measure = predictor.household("TEHC_ST")
        if measure.value is not None:
            return f"{int(measure.value):02d}"
        return hm.measurement_class(measure)

    return {
        "households": len(predictors),
        "admitted_households": len(admitted),
        "predictor_status_order_exclusive": list(PREDICTOR_STATUS_ORDER),
        "predictor_status_exclusive": _counter(p.status for p in predictors),
        "predictor_status_exclusive_admitted": _counter(p.status for p in admitted),
        "predictor_field_classes": {
            name: _counter(hm.measurement_class(p.household(name)) for p in predictors)
            for name in HOUSEHOLD_RESOLVED_FIELDS
        },
        "predictor_field_quality": {
            name: _counter(
                hm.measurement_quality(p.household(name)) for p in predictors
            )
            for name in HOUSEHOLD_RESOLVED_FIELDS
        },
        "predictor_copy_conflicts": {
            name: sum(
                "household_copy_conflict" in p.household(name).reasons
                for p in predictors
            )
            for name in HOUSEHOLD_RESOLVED_FIELDS
        },
        "predictor_unknown_measurements": {
            name: sum(p.household(name).value is None for p in predictors)
            for name in HOUSEHOLD_RESOLVED_FIELDS
        },
        "person_age_measurements": sum(
            len(p.person_age_measurements) for p in predictors
        ),
        "person_age_admissible_measurements": sum(
            _age_admissible(a) for p in predictors for a in p.person_age_measurements
        ),
        "state_of_residence_codes": _counter(_state_label(p) for p in predictors),
        "state_published_non_state_area_households": sum(
            bool(p.diagnostics["state_is_published_non_state_area"]) for p in predictors
        ),
        "state_code_count_published": len(_STATE_CODES),
        "state_codes_that_are_not_a_state_or_dc": list(NON_STATE_AREA_CODES),
        "household_size_agreement": _counter(
            str(p.diagnostics["household_size_agreement"]) for p in predictors
        ),
        "household_size_difference_counts": _counter(
            str(p.diagnostics["household_size_difference"])
            for p in predictors
            if p.diagnostics["household_size_difference"] is not None
        ),
        "children_under_18_agreement": _counter(
            str(p.diagnostics["children_under_18_agreement"]) for p in predictors
        ),
        "age_coverage": _counter(
            str(p.diagnostics["age_coverage"]) for p in predictors
        ),
        "maximum_age_lower_bound_households": sum(
            bool(p.diagnostics["maximum_age_is_lower_bound"]) for p in predictors
        ),
        "maximum_age_at_domain_maximum_households": sum(
            bool(p.diagnostics["maximum_age_at_domain_maximum"]) for p in predictors
        ),
        "maximum_age_unknown_households": sum(
            p.maximum_member_age.value is None for p in predictors
        ),
        "income_sign": _counter(str(p.diagnostics["income_sign"]) for p in predictors),
        "income_at_domain_minimum_households": sum(
            bool(p.diagnostics["income_at_domain_minimum"]) for p in predictors
        ),
        "income_at_domain_maximum_households": sum(
            bool(p.diagnostics["income_at_domain_maximum"]) for p in predictors
        ),
        "eehc_rentsub_state_arm": _counter(
            str(p.diagnostics["eehc_rentsub_state_arm"]) for p in predictors
        ),
        "eehc_rentsub_state_arm_changes_no_classification": True,
        "separation_from_the_measurement_ladder": {
            "exclusion_order_exclusive": list(hm.EXCLUSION_ORDER),
            "predictor_states_added_to_the_exclusion_order": [],
            "predictor_values_read_by_the_exclusion_rule": [],
            "predictors_computed_after_admission": True,
            "admission_counts_preserved_claim": (
                "The exclusion rule's inputs are tenure, assistance, rent and "
                "weight only, and this pass runs after it, so no predictor "
                "value can reach it. That is a structural statement about the "
                "call graph, and the test suite still checks it against a "
                "fixture set measured with and without these columns rather "
                "than asserting it about a real extraction that has not run."
            ),
        },
    }
