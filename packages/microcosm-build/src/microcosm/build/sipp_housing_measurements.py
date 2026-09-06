"""Pure December housing measurements over explicitly supplied 2023 SIPP rows.

Additive to :mod:`microcosm.build.sipp_measurements`, which stays byte-identical.
The frozen sixteen-field December producer binds that module's bytes into its
published producer binding, so a housing field cannot be added to its closed
``_HOUSEHOLD`` registry without moving receipts that already exist. This module
therefore declares its own housing contracts and reuses that module's lexical,
flag, raw-evidence, roster and grouping internals instead of restating them.

No source loading, authentication, fitting, recipient mapping, annualization,
price rebasing or population transfer occurs here. Amounts are December 2022
monthly dollars in their original basis. Categories keep their raw code, their
transcribed allocation flag and the existing ``_QUALITY`` vocabulary; an
out-of-universe answer never becomes an observed negative. Every field contract
below was transcribed from the pinned 2023 SIPP data dictionary; see
docs/us-sipp-december-housing-projection.md for the exact page locators.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from . import sipp_measurements as m

HOUSING_RULE_ID = "sipp_december_housing_measurements_v1"
PersonKey = m.PersonKey

# The dictionary documents no numeric missing sentinel for these amounts: its
# "Not in Universe" note (zero-based page 7) is explicit that '.N'/'N' describe
# internal SIPP data, not the public pipe-delimited file. ``_number`` requires a
# sentinel, so one outside every transcribed domain is supplied and its
# ``missing_item`` class is relabelled ``invalid_domain``. An undocumented token
# must never be reported as though the dictionary defined it as missing.
_UNDOCUMENTED_SENTINEL = -10_000_000

TENURE_OWNED = 1
TENURE_RENTED = 2
TENURE_NO_PAYMENT = 3
YES = 1
NO = 2


class HousingMeasurementError(m.MeasurementError):
    """A requested housing measurement cannot be supplied under this contract."""


@dataclass(frozen=True)
class _Amount:
    """A transcribed December housing amount, with its dictionary locator."""

    flag: str | None
    low: int
    high: int
    grain: str
    concept: str
    basis: str
    universe: str
    universe_description: str
    page: int
    unit: str = "USD_nominal_2022_monthly"


@dataclass(frozen=True)
class _Category:
    """A transcribed categorical status: raw code, flag and strict code domain.

    ``_Field.routes`` in the frozen kernel expresses a yes/no gate and its own
    flag, not a code domain, which is why this separate small kind exists. The
    ``applies_when`` name records the *published* universe gate this lane can
    evaluate from the projection, so an out-of-universe answer is classified
    rather than read as a negative.
    """

    flag: str
    codes: tuple[int, ...]
    labels: tuple[str, ...]
    grain: str
    concept: str
    universe: str
    universe_description: str
    page: int


_HOUSEHOLD_AMOUNTS: Mapping[str, _Amount] = {
    "TRENTMORT": _Amount(
        "ARENTMORT",
        0,
        9_999_999,
        "household",
        "amount_the_household_paid_for_rent_or_mortgage_in_December",
        "December_2022_household_rent_or_mortgage_paid",
        "ETENURE = 2 or (ETENURE = 1 and (EPRDEBT = 1 or EMHDEBT = 1))",
        "Households that either rented their home or owned their home and had "
        "debt against the property.",
        379,
    ),
    "TUTILS": _Amount(
        "AUTILS",
        0,
        9_999_999,
        "household",
        "amount_the_household_paid_for_basic_utilities_in_December",
        "December_2022_household_basic_utilities_paid",
        "ETENURE in (1,2,3)",
        "Households that owned their home, rented their home, or occupied their "
        "home without payment of rent.",
        380,
    ),
}

# Topcode companions carry "Status Flag: NA" and File: Person in the dictionary.
# A populated value marks that person's record as belonging to a topcoded group;
# it is a marker, not the topcode threshold, and no threshold is inferred here.
_PERSON_TOPCODE_AMOUNTS: Mapping[str, _Amount] = {
    "TRNTMT_MED": _Amount(
        None,
        1,
        9_999_999,
        "person",
        "median_of_topcoded_values_for_TRENTMORT_by_demographic_group",
        "topcode_group_median",
        "NA",
        "Amount household paid for rent or mortgage was topcoded.",
        388,
    ),
    "TRNTMT_STD": _Amount(
        None,
        1,
        9_999_999,
        "person",
        "standard_deviation_of_topcoded_values_for_TRENTMORT_by_demographic_group",
        "topcode_group_standard_deviation",
        "NA",
        "Amount household paid for rent or mortgage was topcoded.",
        387,
    ),
    "TUTILS_MED": _Amount(
        None,
        0,
        9_999_999,
        "person",
        "median_of_topcoded_values_for_TUTILS_by_demographic_group",
        "topcode_group_median",
        "NA",
        "Amount household paid for basic utilities was topcoded.",
        389,
    ),
    "TUTILS_STD": _Amount(
        None,
        0,
        9_999_999,
        "person",
        "standard_deviation_of_topcoded_values_for_TUTILS_by_demographic_group",
        "topcode_group_standard_deviation",
        "NA",
        "Amount household paid for basic utilities was topcoded.",
        390,
    ),
}

_TENURE_LABELS = (
    "Owned or being bought by someone in the household?",
    "Rented?",
    "Occupied without payment of rent?",
)
_YES_NO_LABELS = ("Yes", "No")

_HOUSEHOLD_CATEGORIES: Mapping[str, _Category] = {
    "ETENURE": _Category(
        "ATENURE",
        (1, 2, 3),
        _TENURE_LABELS,
        "household",
        "are_living_quarters_owned_rented_or_occupied_without_payment_of_rent",
        "THHLDSTATUS in (1,2,3,4)",
        "All persons",
        784,
    ),
    "ERENTSUB": _Category(
        "ARENTSUB",
        (1, 2),
        _YES_NO_LABELS,
        "household",
        "is_rent_amount_lower_because_of_a_government_housing_program",
        "ETENURE in (2,3)",
        "Respondents who report renting or occupying the living quarters "
        "without payment of rent.",
        788,
    ),
    "EVOUCHER": _Category(
        "AVOUCHER",
        (1, 2),
        _YES_NO_LABELS,
        "household",
        "does_the_household_have_a_housing_voucher",
        "ETENURE in (2,3) and ERENTSUB = 1",
        "Respondents who report the rent amount is lower because they are in a "
        "federal, state or local government housing program.",
        789,
    ),
    "EPRDEBT": _Category(
        "APRDEBT",
        (1, 2),
        _YES_NO_LABELS,
        "household",
        "any_mortgages_or_loans_against_the_primary_residence_in_December",
        "TPRVAL > 0",
        "Households that lived in a non-mobile home and owned their home as of "
        "the last day of the reference period.",
        429,
    ),
    "EMHDEBT": _Category(
        "AMHDEBT",
        (1, 2),
        _YES_NO_LABELS,
        "household",
        "any_mortgages_or_loans_against_the_mobile_home_in_December",
        "TMHVAL > 0",
        "Households that lived in a mobile home and owned their home as of the "
        "last day of the reference period.",
        367,
    ),
}

# Person-month categories. EEHC_RENTSUB is gated by EEHC_TEN, a person-month
# tenure variable distinct from the household ETENURE, plus a residence-state
# condition this projection does not carry. They are recorded separately and
# never determine a household's assistance class.
_PERSON_MONTH_CATEGORIES: Mapping[str, _Category] = {
    "EEHC_TEN": _Category(
        "AEHC_TEN",
        (1, 2, 3),
        _TENURE_LABELS,
        "person_month",
        "tenure_of_the_residence",
        "TPRVLVQRT IN (1,2)",
        "Respondents living in a house/apartment/flat, mobile home/trailer, "
        "rooming house/hotel/motel, or other housing unit",
        786,
    ),
    "EEHC_RENTSUB": _Category(
        "AEHC_RENTSUB",
        (1, 2),
        _YES_NO_LABELS,
        "person_month",
        "rent_subsidy_receipt",
        "EEHC_TEN in (2,3) and TEHC_ST <= '60'",
        "Respondents renting or occupying without payment of rent and report "
        "the US as country of residence",
        790,
    ),
    "EEHC_VOUCHER": _Category(
        "AEHC_VOUCHER",
        (1, 2),
        _YES_NO_LABELS,
        "person_month",
        "housing_voucher_receipt",
        "EEHC_RENTSUB = 1",
        "Respondents who reported receiving a rent subsidy",
        791,
    ),
}

HOUSEHOLD_AMOUNTS = tuple(_HOUSEHOLD_AMOUNTS)
HOUSEHOLD_CATEGORIES = tuple(_HOUSEHOLD_CATEGORIES)
PERSON_MONTH_CATEGORIES = tuple(_PERSON_MONTH_CATEGORIES)
PERSON_TOPCODE_AMOUNTS = tuple(_PERSON_TOPCODE_AMOUNTS)

# Exclusive exclusion ladder for the unassisted-renter donor universe. The order
# is a declared convention so a household reports one reason; every non-exclusive
# condition is still counted separately in the diagnostics.
EXCLUSION_ORDER = (
    "tenure_copy_conflict",
    "tenure_unknown",
    "tenure_owner",
    "tenure_no_payment",
    "assistance_universe_conflict",
    "assistance_copy_conflict",
    "assistance_assisted",
    "assistance_unassisted_allocated",
    "assistance_unknown",
    "rent_copy_conflict",
    "rent_unknown",
    "weight_unknown",
    "weight_not_positive",
)


def _amount_lexical(
    values: Mapping, name: str, spec: _Amount
) -> tuple[str, Decimal | None]:
    lexical, value = m._number(
        values, name, spec.low, spec.high, _UNDOCUMENTED_SENTINEL
    )
    return ("invalid_domain" if lexical == "missing_item" else lexical), value


def _observe_amount(
    key: PersonKey, month: int, row: m.SippRow, name: str, spec: _Amount
) -> m.Observation:
    lexical, value = _amount_lexical(row.values, name, spec)
    reasons = list(row.contradictions)
    if value is None:
        reasons.append(f"{name}:{lexical}")
    if spec.flag is None:
        flag, quality = None, "no_published_status_flag"
        origin = "published_topcode_marker"
    else:
        flag, quality = m._flag(row.values, spec.flag)
        origin = "published_source_value"
        if flag is None or flag <= 0:
            reasons.append(f"{spec.flag}:{quality}")
    raw = (name,) if spec.flag is None else (name, spec.flag)
    return m.Observation(
        key,
        month,
        name,
        value,
        lexical,
        origin,
        flag,
        quality,
        m._raw(row.values, raw),
        tuple(dict.fromkeys(reasons)),
    )


def _observe_category(
    key: PersonKey, month: int, row: m.SippRow, name: str, spec: _Category
) -> m.Observation:
    low, high = min(spec.codes), max(spec.codes)
    lexical, value = m._number(row.values, name, low, high, _UNDOCUMENTED_SENTINEL)
    if lexical == "missing_item":
        lexical = "invalid_domain"
    if value is not None and int(value) not in spec.codes:
        # Strict set membership: ``_number`` only bounds a range, and no housing
        # code domain may be widened by a future gap inside its own bounds.
        lexical, value = "invalid_domain", None
    flag, quality = m._flag(row.values, spec.flag)
    reasons = list(row.contradictions)
    if value is None:
        reasons.append(f"{name}:{lexical}")
    if flag is None or flag <= 0:
        reasons.append(f"{spec.flag}:{quality}")
    return m.Observation(
        key,
        month,
        name,
        value,
        lexical,
        "published_source_category",
        flag,
        quality,
        m._raw(row.values, (name, spec.flag)),
        tuple(dict.fromkeys(reasons)),
    )


def _build(
    name: str,
    concept: str,
    source: m.SippSource,
    basis: str,
    unit: str,
    observations: tuple[m.Observation, ...],
    value: Decimal | None,
    status: str,
    reasons: Iterable[str] = (),
) -> m.Measurement:
    return m.Measurement(
        name,
        concept,
        source,
        basis,
        unit,
        value,
        status,
        observations,
        tuple(dict.fromkeys(reasons)),
        HOUSING_RULE_ID,
    )


def _copied(
    name: str,
    concept: str,
    source: m.SippSource,
    basis: str,
    unit: str,
    observations: tuple[m.Observation, ...],
) -> m.Measurement:
    """Apply the frozen kernel's copy-agreement rule to a household-copied field."""
    signatures = {(o.value, o.lexical_class, o.flag, o.quality) for o in observations}
    reasons = [r for o in observations for r in o.reasons]
    if len(signatures) > 1:
        reasons.append("household_copy_conflict")
    value = observations[0].value if not reasons else None
    return _build(
        name,
        concept,
        source,
        basis,
        unit,
        observations,
        value,
        "household_copy_agreement" if value is not None else "unknown_household_copy",
        reasons,
    )


def measurement_class(measure: m.Measurement) -> str:
    """Classify a measurement's evidence without collapsing NIU into a negative."""
    if "household_copy_conflict" in measure.reasons:
        return "copy_conflict"
    flags = tuple(o.flag for o in measure.observations)
    if all(o.quality == "no_published_status_flag" for o in measure.observations):
        # The topcode companions carry "Status Flag: NA" in the dictionary; a
        # published value there cannot be graded against an allocation code.
        return (
            "observed_without_status_flag" if measure.value is not None else "unknown"
        )
    if measure.value is not None and not measure.reasons:
        if all(f == 1 for f in flags):
            return "observed"
        return "allocated"
    if flags and all(f == 0 for f in flags):
        return "not_in_universe"
    if any(f == -99 for f in flags):
        return "suppressed"
    return "unknown"


def measurement_quality(measure: m.Measurement) -> str:
    """Report the allocation class, keeping reported distinct from every imputation."""
    qualities = {o.quality for o in measure.observations}
    if len(qualities) == 1:
        return next(iter(qualities))
    return "mixed_source_quality"


@dataclass(frozen=True)
class HousingHousehold:
    """One December residence's housing evidence; no fit admission is implied."""

    key: tuple[str, int]
    members: tuple[PersonKey, ...]
    weight: m.Measurement
    person_weights: tuple[m.Measurement, ...]
    household_measurements: tuple[m.Measurement, ...]
    person_month_measurements: tuple[m.Measurement, ...]
    person_topcode_measurements: tuple[m.Measurement, ...]
    tenure: str
    assistance: str
    rent: m.Measurement
    utilities: m.Measurement
    exclusion: str | None
    diagnostics: Mapping[str, object]

    def household(self, name: str) -> m.Measurement:
        for measure in self.household_measurements:
            if measure.concept == name:
                return measure
        raise HousingMeasurementError("UNKNOWN_HOUSEHOLD_FIELD:" + name)

    def person_month(self, name: str) -> tuple[m.Measurement, ...]:
        return tuple(v for v in self.person_month_measurements if v.concept == name)

    def person_topcode(self, name: str) -> tuple[m.Measurement, ...]:
        return tuple(v for v in self.person_topcode_measurements if v.concept == name)


def _tenure_class(measure: m.Measurement) -> str:
    state = measurement_class(measure)
    if state in ("observed", "allocated"):
        return {
            TENURE_OWNED: "owner",
            TENURE_RENTED: "renter",
            TENURE_NO_PAYMENT: "no_payment",
        }[int(measure.value)]
    return "tenure_" + state


def _applicable(measure: m.Measurement) -> bool:
    """A published answer inside its own universe: a value with an applicable flag."""
    return measurement_class(measure) in ("observed", "allocated")


def _assistance_class(
    tenure: str, rentsub: m.Measurement, voucher: m.Measurement
) -> str:
    """Classify household assistance strictly; never recode unknown as unassisted."""
    rentsub_state = measurement_class(rentsub)
    voucher_applicable = _applicable(voucher)
    if tenure == "owner":
        # ERENTSUB's published universe is ETENURE in (2,3).
        if rentsub_state in ("observed", "allocated") or voucher_applicable:
            return "assistance_universe_conflict"
        return "not_in_assistance_universe"
    if tenure not in ("renter", "no_payment"):
        return "assistance_unknown"
    if rentsub_state == "copy_conflict":
        return "assistance_copy_conflict"
    if rentsub_state == "not_in_universe":
        # ERENTSUB's published universe already contains this tenure, so a
        # not-in-universe flag is retained as its own class rather than merged
        # into plain missingness, and never read as a negative answer.
        return (
            "assistance_universe_conflict"
            if voucher_applicable
            else "assistance_not_in_universe"
        )
    if rentsub_state in ("observed", "allocated"):
        code = int(rentsub.value)
        if code == YES:
            if voucher_applicable:
                return (
                    "assisted_with_voucher"
                    if int(voucher.value) == YES
                    else "assisted_without_voucher"
                )
            return "assisted_voucher_unknown"
        # EVOUCHER's published universe requires ERENTSUB = 1; an applicable
        # answer alongside an explicit "No" contradicts the published universe.
        if voucher_applicable:
            return "assistance_universe_conflict"
        # An imputed "No" is not an explicit negative answer.
        return (
            "explicit_unassisted"
            if rentsub_state == "observed"
            else "unassisted_allocated"
        )
    if voucher_applicable:
        return "assistance_universe_conflict"
    return "assistance_unknown"


def _rent_measurement(
    source: m.SippSource,
    tenure: str,
    amount: m.Measurement,
    tenure_measure: m.Measurement,
) -> m.Measurement:
    """Admit TRENTMORT as contract rent paid only on the renter tenure universe.

    An owner household's TRENTMORT is a mortgage payment and is refused with an
    explicit reason; it is never converted into a zero rent. ETENURE = 3 is
    outside TRENTMORT's published universe, so it is refused too.
    """
    spec = _HOUSEHOLD_AMOUNTS["TRENTMORT"]
    reasons: list[str] = []
    if tenure != "renter":
        reasons.append("tenure_not_renter:" + tenure)
        if tenure == "owner" and amount.value is not None:
            reasons.append("mortgage_excluded")
        if tenure == "no_payment" and amount.value is not None:
            reasons.append("no_payment_tenure_outside_trentmort_universe")
    reasons.extend(amount.reasons)
    if amount.value is None and not amount.reasons:
        reasons.append("TRENTMORT:unknown")
    value = amount.value if not reasons else None
    return _build(
        "sipp_december_renter_contract_rent_paid",
        "TRENTMORT_on_renter_tenure",
        source,
        "December_2022_monthly_contract_rent_paid_by_the_household",
        spec.unit,
        amount.observations + tenure_measure.observations,
        value,
        "renter_contract_rent_paid" if value is not None else "unknown_renter_rent",
        reasons,
    )


def _topcode_marked(measures: Sequence[m.Measurement]) -> tuple[int, int]:
    marked = sum(v.value is not None for v in measures)
    return marked, len(measures)


def _exclusion(
    tenure: str, assistance: str, rent: m.Measurement, weight: m.Measurement
) -> str | None:
    if tenure == "tenure_copy_conflict":
        return "tenure_copy_conflict"
    if tenure not in ("owner", "renter", "no_payment"):
        return "tenure_unknown"
    if tenure == "owner":
        return "tenure_owner"
    if tenure == "no_payment":
        return "tenure_no_payment"
    if assistance == "assistance_universe_conflict":
        return "assistance_universe_conflict"
    if assistance == "assistance_copy_conflict":
        return "assistance_copy_conflict"
    if assistance.startswith("assisted"):
        return "assistance_assisted"
    if assistance == "unassisted_allocated":
        return "assistance_unassisted_allocated"
    if assistance != "explicit_unassisted":
        return "assistance_unknown"
    if "household_copy_conflict" in rent.reasons:
        return "rent_copy_conflict"
    if rent.value is None:
        return "rent_unknown"
    if weight.value is None:
        return "weight_unknown"
    if weight.value <= 0:
        return "weight_not_positive"
    return None


def december_housing(
    rows: Sequence[m.SippRow],
    *,
    source: m.SippSource,
    complete_membership: bool,
    target_year: int = 2022,
) -> tuple[HousingHousehold, ...]:
    """Measure December housing on the frozen roster, keeping every grain distinct.

    The authoritative roster, residence grouping and weights come from
    ``sipp_measurements.december_households``; this function regroups the same
    supplied rows and refuses if the two disagree, so the housing lane can never
    quietly adopt a different household identity. ``complete_membership=True``
    remains a caller assertion, not independent authentication.
    """
    m._check_source(source, target_year)
    roster = m.december_households(
        rows, source=source, complete_membership=complete_membership
    )
    fields = (
        *_HOUSEHOLD_AMOUNTS,
        *(v.flag for v in _HOUSEHOLD_AMOUNTS.values() if v.flag),
        *_HOUSEHOLD_CATEGORIES,
        *(v.flag for v in _HOUSEHOLD_CATEGORIES.values()),
        *_PERSON_MONTH_CATEGORIES,
        *(v.flag for v in _PERSON_MONTH_CATEGORIES.values()),
        *_PERSON_TOPCODE_AMOUNTS,
        "ERESIDENCEID",
        "WPFINWGT",
    )
    prepared = m._prepare(rows, fields)
    groups: dict[tuple[str, int], list[tuple[PersonKey, m.SippRow]]] = {}
    for key, month, row in prepared:
        if month != 12:
            continue
        _, residence = m._number(row.values, "ERESIDENCEID", 100001, 400999, -999)
        if residence is None:
            raise HousingMeasurementError("INVALID_DECEMBER_RESIDENCE")
        groups.setdefault((key[0], int(residence)), []).append((key, row))
    regrouped = tuple(
        (key, tuple(person for person, _ in members))
        for key, members in sorted(groups.items())
    )
    if regrouped != tuple((h.key, h.members) for h in roster):
        raise HousingMeasurementError("HOUSING_ROSTER_DISAGREEMENT")
    result = []
    for household in roster:
        members = groups[household.key]
        result.append(_household(source, household, members))
    return tuple(result)


def _household(
    source: m.SippSource,
    household: m.DecemberHousehold,
    members: Sequence[tuple[PersonKey, m.SippRow]],
) -> HousingHousehold:
    household_measures = []
    for name, spec in _HOUSEHOLD_CATEGORIES.items():
        observations = tuple(
            _observe_category(key, 12, row, name, spec) for key, row in members
        )
        household_measures.append(
            _copied(
                "sipp_december_" + name.lower(),
                name,
                source,
                "December_2022_household_category",
                "code",
                observations,
            )
        )
    for name, spec in _HOUSEHOLD_AMOUNTS.items():
        observations = tuple(
            _observe_amount(key, 12, row, name, spec) for key, row in members
        )
        household_measures.append(
            _copied(
                "sipp_december_" + name.lower(),
                name,
                source,
                spec.basis,
                spec.unit,
                observations,
            )
        )
    person_month = []
    for name, spec in _PERSON_MONTH_CATEGORIES.items():
        for key, row in members:
            observation = _observe_category(key, 12, row, name, spec)
            person_month.append(
                _build(
                    "sipp_december_" + name.lower(),
                    name,
                    source,
                    "December_2022_person_month_category",
                    "code",
                    (observation,),
                    observation.value if not observation.reasons else None,
                    "published_person_month_category"
                    if observation.usable
                    else "unknown_person_month_category",
                    observation.reasons,
                )
            )
    topcodes = []
    for name, spec in _PERSON_TOPCODE_AMOUNTS.items():
        for key, row in members:
            observation = _observe_amount(key, 12, row, name, spec)
            topcodes.append(
                _build(
                    "sipp_december_" + name.lower(),
                    name,
                    source,
                    spec.basis,
                    spec.unit,
                    (observation,),
                    observation.value,
                    "topcode_group_marker"
                    if observation.value is not None
                    else "no_topcode_group_marker",
                    observation.reasons if observation.value is None else (),
                )
            )
    lookup = {v.concept: v for v in household_measures}
    tenure = _tenure_class(lookup["ETENURE"])
    assistance = _assistance_class(tenure, lookup["ERENTSUB"], lookup["EVOUCHER"])
    rent = _rent_measurement(source, tenure, lookup["TRENTMORT"], lookup["ETENURE"])
    utilities = lookup["TUTILS"]
    exclusion = _exclusion(tenure, assistance, rent, household.weight)
    return HousingHousehold(
        household.key,
        household.members,
        household.weight,
        household.person_weights,
        tuple(household_measures),
        tuple(person_month),
        tuple(topcodes),
        tenure,
        assistance,
        rent,
        utilities,
        exclusion,
        _diagnostics(lookup, person_month, topcodes, tenure, rent, utilities),
    )


def _person_month_codes(
    measures: Sequence[m.Measurement], name: str
) -> tuple[dict[str, int], bool]:
    selected = [v for v in measures if v.concept == name]
    codes: dict[str, int] = {}
    for measure in selected:
        label = (
            str(int(measure.value))
            if measure.value is not None
            else measurement_class(measure)
        )
        codes[label] = codes.get(label, 0) + 1
    known = {k for k in codes if k.isdigit()}
    return dict(sorted(codes.items())), len(known) > 1


def _diagnostics(
    lookup: Mapping[str, m.Measurement],
    person_month: Sequence[m.Measurement],
    topcodes: Sequence[m.Measurement],
    tenure: str,
    rent: m.Measurement,
    utilities: m.Measurement,
) -> Mapping[str, object]:
    rent_amount = lookup["TRENTMORT"]
    rent_marked, rent_total = _topcode_marked(
        [v for v in topcodes if v.concept in ("TRNTMT_MED", "TRNTMT_STD")]
    )
    util_marked, util_total = _topcode_marked(
        [v for v in topcodes if v.concept in ("TUTILS_MED", "TUTILS_STD")]
    )
    owner_debt = "not_applicable"
    if tenure == "owner":
        debts = [_applicable(lookup["EPRDEBT"]), _applicable(lookup["EMHDEBT"])]
        values = [
            int(lookup[n].value)
            for n in ("EPRDEBT", "EMHDEBT")
            if _applicable(lookup[n])
        ]
        if not any(debts):
            owner_debt = "owner_debt_unresolved"
        elif YES in values:
            owner_debt = "owner_debt_reported"
        else:
            owner_debt = "owner_debt_absent"
    eehc_rentsub_codes, eehc_rentsub_disagreement = _person_month_codes(
        person_month, "EEHC_RENTSUB"
    )
    rentsub_receipt_members = sum(
        v.concept == "EEHC_RENTSUB" and v.value is not None and int(v.value) == YES
        for v in person_month
    )
    eehc_voucher_codes, eehc_voucher_disagreement = _person_month_codes(
        person_month, "EEHC_VOUCHER"
    )
    eehc_ten_codes, eehc_ten_disagreement = _person_month_codes(
        person_month, "EEHC_TEN"
    )
    return {
        "tenure_class": measurement_class(lookup["ETENURE"]),
        "tenure_quality": measurement_quality(lookup["ETENURE"]),
        "rentsub_class": measurement_class(lookup["ERENTSUB"]),
        "rentsub_quality": measurement_quality(lookup["ERENTSUB"]),
        "voucher_class": measurement_class(lookup["EVOUCHER"]),
        "voucher_quality": measurement_quality(lookup["EVOUCHER"]),
        "rent_amount_class": measurement_class(rent_amount),
        "rent_amount_quality": measurement_quality(rent_amount),
        "utilities_class": measurement_class(utilities),
        "utilities_quality": measurement_quality(utilities),
        "owner_debt": owner_debt,
        "rent_admitted": rent.value is not None,
        "rent_zero": rent.value is not None and rent.value == 0,
        "rent_at_domain_maximum": rent.value is not None
        and rent.value == _HOUSEHOLD_AMOUNTS["TRENTMORT"].high,
        "utilities_zero": utilities.value is not None and utilities.value == 0,
        "utilities_at_domain_maximum": utilities.value is not None
        and utilities.value == _HOUSEHOLD_AMOUNTS["TUTILS"].high,
        "rent_topcode_marked_records": rent_marked,
        "rent_topcode_marker_records": rent_total,
        "rent_topcode_partial_marking": 0 < rent_marked < rent_total,
        "utilities_topcode_marked_records": util_marked,
        "utilities_topcode_marker_records": util_total,
        "utilities_topcode_partial_marking": 0 < util_marked < util_total,
        "eehc_rentsub_receipt_members": rentsub_receipt_members,
        "eehc_rentsub_member_codes": eehc_rentsub_codes,
        "eehc_rentsub_member_disagreement": eehc_rentsub_disagreement,
        "eehc_voucher_member_codes": eehc_voucher_codes,
        "eehc_voucher_member_disagreement": eehc_voucher_disagreement,
        "eehc_ten_member_codes": eehc_ten_codes,
        "eehc_ten_member_disagreement": eehc_ten_disagreement,
    }
