"""Pure, additive measurements of explicitly supplied 2023 SIPP source rows.

No source loading, authentication, fitting, model calculations or year transfer occurs
here. The caller retains original observations and authenticates the supplied source
identity/complete December roster. Results retain only requested field evidence.
See docs/us-sipp-measurements.md for source pages, required fields and boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from types import MappingProxyType

SIPP_2023_MASTER_SHA256 = (
    "b62876a0763a0b45f2c4873b055b35c264d3a67640b8a8b3590b650aa1db679e"
)
MEASUREMENT_RULE_ID = "sipp_source_measurements_v1"
PersonKey = tuple[str, int]
RawValue = str | None


class MeasurementError(ValueError):
    """A requested measurement cannot be supplied under this source contract."""


@dataclass(frozen=True)
class SippSource:
    """Caller-supplied identity, not proof that rows came from authenticated bytes."""

    data_sha256: str
    metadata_sha256: str
    survey_year: int
    reference_year: int


@dataclass(frozen=True)
class SippRow:
    """Lexical source fields plus explicit contradictions found by the caller.

    Values must be strings or nulls; absent keys and lexical blanks are distinct.
    Callers must surface additional contradictory evidence rather than omit it from
    a projection. Operations snapshot required fields and do not retain this mapping.
    """

    values: Mapping[str, RawValue]
    contradictions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawField:
    """Small copied field evidence; presence distinguishes absent columns from nulls."""

    name: str
    present: bool
    value: RawValue
    input_type: str


@dataclass(frozen=True)
class Observation:
    """One requested field's lexical, value, allocation and route evidence."""

    person: PersonKey
    month: int
    field: str
    value: Decimal | None
    lexical_class: str
    origin: str
    flag: int | None
    quality: str
    raw_fields: tuple[RawField, ...]
    reasons: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.value is not None and not self.reasons


@dataclass(frozen=True)
class Measurement:
    """A source-specific value or explicit unknown; no fit admission is implied."""

    name: str
    concept: str
    source: SippSource
    basis: str
    unit: str
    value: Decimal | None
    status: str
    observations: tuple[Observation, ...]
    reasons: tuple[str, ...] = ()
    rule_id: str = MEASUREMENT_RULE_ID

    def require_value(self) -> Decimal:
        """Refuse unknown measurements before a later strict adapter uses them."""
        if self.value is None:
            raise MeasurementError("UNKNOWN_MEASUREMENT:" + self.name)
        return self.value


@dataclass(frozen=True)
class DecemberHousehold:
    """Full supplied December membership and measurements, before target selection."""

    key: tuple[str, int]
    members: tuple[PersonKey, ...]
    weight: Measurement
    person_weights: tuple[Measurement, ...]
    measurements: tuple[Measurement, ...]


@dataclass(frozen=True)
class _Field:
    flag: str
    low: int
    high: int
    missing: int
    basis: str
    unit: str = "USD_nominal_2022"
    routes: tuple[tuple[str, str], ...] = ()
    extra: tuple[str, ...] = ()


# Closed small field contracts, transcribed from the pinned master and dictionary.
# These are source measurements, not a general registry or a recipient mapping.
_EARNINGS = _Field(
    "APEARN_ALT", -99999999, 99999999, -100000998, "alternate_equal_month"
)
_RETIREMENT = _Field(
    "ARETINCAMT", 0, 10000000000, -999, "monthly_broad_retirement_recode"
)
_ANNUAL = {
    "TLIFEAMT": _Field(
        "ALIFEAMT",
        1,
        10000000,
        -999,
        "annual_once",
        routes=(("ELIFEYN", "ALIFEYN"),),
        extra=("ELIFEREG", "ALIFEREG"),
    ),
    "TLMPAMT": _Field(
        "ALMPAMT",
        1,
        10000000,
        -999,
        "annual_once",
        routes=(
            ("ELMPTYP1YN", "ALMPTYPYN"),
            ("ELMPTYP2YN", "ALMPTYPYN"),
            ("ELMPTYP4YN", "ALMPTYPYN"),
        ),
    ),
    "TROLLAMT": _Field(
        "AROLLAMT",
        1,
        10000000,
        -999,
        "annual_once",
        routes=(("EROLLOVR1", "AROLLOVR1"), ("EROLLOVR2", "AROLLOVR2")),
    ),
    "TIRA_INC_AMT": _Field(
        "AIRA_INC_AMT",
        0,
        9999999,
        -999,
        "annual_once",
        routes=(("EIRA_INC_YN", "AIRA_INC_YN"),),
    ),
    "TTHR_INC_AMT": _Field(
        "ATHR_INC_AMT",
        0,
        9999999,
        -999,
        "annual_once",
        routes=(("ETHR_INC_YN", "ATHR_INC_YN"),),
    ),
}
_HOUSEHOLD = {
    "TVEH_NUM": _Field(
        "AVEH_NUM", 0, 20, -999, "reference_period_owned_cars_trucks_vans", "count"
    ),
    "THVAL_VEH": _Field(
        "AHVAL_VEH", 0, 99999999, -999, "reference_year_end_all_vehicle_value"
    ),
    "THEQ_VEH": _Field(
        "AHEQ_VEH",
        -9999999,
        9999999,
        -10000998,
        "reference_year_end_all_vehicle_equity",
    ),
    "THDEBT_VEH": _Field(
        "AHDEBT_VEH", 0, 99999999, -999, "reference_year_end_all_vehicle_debt"
    ),
}
_QUALITY = {
    -99: "suppressed",
    0: "not_in_universe",
    1: "reported",
    2: "hot_deck_imputed",
    3: "logically_imputed",
    4: "model_imputed",
    5: "cold_deck_imputed",
    6: "range_imputed",
    7: "reported_and_imputed_components",
    8: "imputed_components",
    9: "component_derived_allocation_unresolved",
}
_KEYS = ("SSUID", "PNUM", "MONTHCODE")
_EARN_CONTROLS = ("RMNUMJOBS", "AMNUMJOBS", "RMESR", "AMESR", "TAGE", "TAGE_EHC")
_OPTIONAL_EARN_EVIDENCE = (
    "TPEARN",
    *(f"EJB{n}_{bound}" for n in range(1, 7) for bound in ("BMONTH", "EMONTH")),
)
_NUMBER = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?\Z")
# Copy this fixed context at each arithmetic operation: caller rounding, traps,
# exponent limits and flags must not affect a source measurement.
_ARITHMETIC_CONTEXT = Context(
    prec=80,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)


def _check_source(source: SippSource, target_year: int) -> None:
    if (
        not isinstance(source, SippSource)
        or not isinstance(source.data_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source.data_sha256)
        or source.metadata_sha256 != SIPP_2023_MASTER_SHA256
        or type(source.survey_year) is not int
        or source.survey_year != 2023
        or type(source.reference_year) is not int
        or source.reference_year != 2022
    ):
        raise MeasurementError("SOURCE_CONTRACT_MISMATCH")
    if type(target_year) is not int or target_year != source.reference_year:
        raise MeasurementError("YEAR_BRIDGE_REQUIRED")


def _number(
    values: Mapping,
    name: str,
    low: int,
    high: int,
    missing: int,
    *,
    integral: bool = True,
) -> tuple[str, Decimal | None]:
    if name not in values:
        return "absent", None
    raw = values[name]
    if raw is None:
        return "null", None
    if type(raw) is not str:
        return "invalid_type", None
    text = raw.strip()
    if not text:
        return "blank", None
    if len(text) > 64 or not _NUMBER.fullmatch(text):
        return "invalid_lexical", None
    value = Decimal(text)
    if value == missing:
        return "missing_item", None
    if (integral and value != value.to_integral_value()) or not low <= value <= high:
        return "invalid_domain", None
    return ("numeric_zero" if value == 0 else "numeric"), value


def _flag(values: Mapping, name: str) -> tuple[int | None, str]:
    lexical, value = _number(values, name, -99, 9, -999)
    if value is None:
        return None, "invalid_status" if lexical.startswith(
            "invalid"
        ) else "missing_status"
    if int(value) not in _QUALITY:
        return None, "invalid_status"
    flag = int(value)
    return flag, _QUALITY[flag]


def _raw(values: Mapping, fields: Iterable[str]) -> tuple[RawField, ...]:
    return tuple(
        RawField(
            f,
            f in values,
            values.get(f) if type(values.get(f)) is str else None,
            type(values[f]).__name__ if f in values else "absent",
        )
        for f in dict.fromkeys(fields)
    )


def _prepare(
    rows: Sequence[SippRow], fields: Iterable[str]
) -> tuple[tuple[PersonKey, int, SippRow], ...]:
    needed = tuple(dict.fromkeys((*_KEYS, *fields)))
    prepared = []
    seen = set()
    for supplied in rows:
        if not isinstance(supplied, SippRow) or not isinstance(
            supplied.values, Mapping
        ):
            raise MeasurementError("INVALID_ROW")
        if type(supplied.contradictions) is not tuple:
            raise MeasurementError("INVALID_CONTRADICTION_EVIDENCE")
        contradictions = supplied.contradictions
        if any(type(r) is not str or not r or len(r) > 128 for r in contradictions):
            raise MeasurementError("INVALID_CONTRADICTION_EVIDENCE")
        row = SippRow(
            MappingProxyType(
                {k: supplied.values[k] for k in needed if k in supplied.values}
            ),
            contradictions,
        )
        ssuid = row.values.get("SSUID")
        if (
            type(ssuid) is not str
            or not ssuid
            or ssuid.strip() != ssuid
            or ssuid == "-999"
            or len(ssuid) > 64
        ):
            raise MeasurementError("INVALID_PERSON_KEY")
        _, person = _number(row.values, "PNUM", 101, 499, -999)
        _, month = _number(row.values, "MONTHCODE", 1, 12, -9)
        if person is None or month is None:
            raise MeasurementError("INVALID_PERSON_MONTH_KEY")
        key = (ssuid, int(person))
        identity = (*key, int(month))
        if identity in seen:
            raise MeasurementError("DUPLICATE_PERSON_MONTH")
        seen.add(identity)
        prepared.append((key, int(month), row))
    return tuple(sorted(prepared, key=lambda p: (*p[0], p[1])))


def _person_rows(
    rows: Sequence[SippRow], fields: Iterable[str]
) -> tuple[tuple[PersonKey, int, SippRow], ...]:
    prepared = _prepare(rows, fields)
    if not prepared or len({p[0] for p in prepared}) != 1:
        raise MeasurementError("EXACTLY_ONE_PERSON_REQUIRED")
    return prepared


def _observe(
    key: PersonKey,
    month: int,
    row: SippRow,
    field: str,
    spec: _Field,
    extra: Iterable[str] = (),
) -> Observation:
    lexical, value = _number(row.values, field, spec.low, spec.high, spec.missing)
    flag, quality = _flag(row.values, spec.flag)
    reasons = list(row.contradictions)
    if value is None:
        reasons.append(f"{field}:{lexical}")
    if flag is None or flag <= 0:
        reasons.append(f"{spec.flag}:{quality}")
    return Observation(
        key,
        month,
        field,
        value,
        lexical,
        "published_source_value",
        flag,
        quality,
        _raw(row.values, (field, spec.flag, *extra)),
        tuple(reasons),
    )


def _earnings_month(key: PersonKey, month: int, row: SippRow) -> Observation:
    extras = (*_EARN_CONTROLS, *(f for f in _OPTIONAL_EARN_EVIDENCE if f in row.values))
    obs = _observe(key, month, row, "TPEARN_ALT", _EARNINGS, extras)
    if obs.lexical_class != "blank" or obs.flag != 0:
        return obs
    contradictions = list(row.contradictions)
    _, parallel = _number(row.values, "TPEARN", -99999999, 99999999, -100000998)
    if parallel is not None and parallel != 0:
        contradictions.append("nonzero_parallel_earnings_recode")
    for n in range(1, 7):
        _, start = _number(row.values, f"EJB{n}_BMONTH", 1, 12, -9)
        _, end = _number(row.values, f"EJB{n}_EMONTH", 1, 12, -9)
        if start is not None and end is not None and start <= month <= end:
            contradictions.append(f"active_public_job:{n}")
    _, jobs = _number(row.values, "RMNUMJOBS", 0, 17, -999)
    _, status = _number(row.values, "RMESR", 1, 8, -9)
    _, age = _number(row.values, "TAGE", 0, 90, -999)
    job_flag, job_quality = _flag(row.values, "AMNUMJOBS")
    status_flag, status_quality = _flag(row.values, "AMESR")
    if (
        jobs == 0
        and status in (6, 7, 8)
        and age is not None
        and age >= 15
        and job_flag is not None
        and job_flag > 0
        and status_flag is not None
        and status_flag > 0
        and not contradictions
    ):
        return replace(
            obs,
            value=Decimal(0),
            origin="structural_zero_no_job_v1",
            quality=job_quality
            if job_quality == status_quality
            else "mixed_source_quality",
            reasons=(),
        )
    return replace(
        obs,
        reasons=tuple(
            dict.fromkeys((*obs.reasons, *contradictions, "unknown_no_job_route"))
        ),
    )


def _measurement(
    name: str,
    concept: str,
    source: SippSource,
    spec: _Field,
    observations: tuple[Observation, ...],
    value: Decimal | None,
    status: str,
    reasons: Iterable[str] = (),
) -> Measurement:
    return Measurement(
        name,
        concept,
        source,
        spec.basis,
        spec.unit,
        value,
        status,
        observations,
        tuple(dict.fromkeys(reasons)),
    )


def _annual_sum(
    observations: tuple[Observation, ...],
    source: SippSource,
    spec: _Field,
    name: str,
    concept: str,
) -> Measurement:
    reasons = [
        f"missing_month:{m}"
        for m in range(1, 13)
        if m not in {o.month for o in observations}
    ]
    reasons.extend(f"month:{o.month}:{r}" for o in observations for r in o.reasons)
    if reasons:
        return _measurement(
            name, concept, source, spec, observations, None, "unknown_annual", reasons
        )
    with localcontext(_ARITHMETIC_CONTEXT):
        value = sum((o.value for o in observations), Decimal(0))
    zeros = sum(o.origin == "structural_zero_no_job_v1" for o in observations)
    status = (
        "complete_all_structural_zero"
        if zeros == 12
        else "complete_with_structural_zero"
        if zeros
        else "complete_published"
    )
    return _measurement(name, concept, source, spec, observations, value, status)


def annual_earnings(
    rows: Sequence[SippRow], *, source: SippSource, target_year: int = 2022
) -> Measurement:
    """Sum a complete person's alternate-basis year, admitting the reviewed zero route."""
    _check_source(source, target_year)
    prepared = _person_rows(
        rows, ("TPEARN_ALT", "APEARN_ALT", *_EARN_CONTROLS, *_OPTIONAL_EARN_EVIDENCE)
    )
    observations = tuple(_earnings_month(*p) for p in prepared)
    return _annual_sum(
        observations,
        source,
        _EARNINGS,
        "sipp_current_total_earnings_alt_annual",
        "current_total_earnings",
    )


def annual_retirement(
    rows: Sequence[SippRow], *, source: SippSource, target_year: int = 2022
) -> Measurement:
    """Sum the broad published recode; never reconstruct or double-add its components."""
    _check_source(source, target_year)
    prepared = _person_rows(rows, ("TRETINCAMT", "ARETINCAMT", "TAGE_EHC"))
    observations = []
    for key, month, row in prepared:
        obs = _observe(key, month, row, "TRETINCAMT", _RETIREMENT, ("TAGE_EHC",))
        _, age = _number(row.values, "TAGE_EHC", 0, 90, -999)
        if obs.value is not None and (age is None or age < 15):
            obs = replace(
                obs, reasons=(*obs.reasons, "retirement_age_universe_unresolved")
            )
        observations.append(obs)
    return _annual_sum(
        tuple(observations),
        source,
        _RETIREMENT,
        "sipp_broad_retirement_recode_annual",
        "sipp_broad_retirement_recode",
    )


def annual_once(
    rows: Sequence[SippRow], field: str, *, source: SippSource, target_year: int = 2022
) -> Measurement:
    """Take one agreeing annual amount; missing repetitions are retained, not filled."""
    _check_source(source, target_year)
    if field not in _ANNUAL:
        raise MeasurementError("UNSUPPORTED_ANNUAL_ONCE_FIELD")
    spec = _ANNUAL[field]
    route_fields = tuple(f for pair in spec.routes for f in pair)
    prepared = _person_rows(rows, (field, spec.flag, *route_fields, *spec.extra))
    observations = []
    route_values = {f: set() for f, _ in spec.routes}
    fatal = []
    for key, month, row in prepared:
        obs = _observe(key, month, row, field, spec, (*route_fields, *spec.extra))
        applicable = False
        for route, flag_field in spec.routes:
            lexical, value = _number(row.values, route, 1, 2, -9)
            flag, flag_quality = _flag(row.values, flag_field)
            if flag_quality == "invalid_status":
                fatal.append(f"month:{month}:{flag_field}:invalid_status")
            if value is not None and flag is not None and flag > 0:
                route_values[route].add(value)
                applicable |= value == 1
            elif lexical.startswith("invalid"):
                fatal.append(f"month:{month}:{route}:{lexical}")
        if obs.value is not None and not applicable:
            obs = replace(obs, reasons=(*obs.reasons, "annual_route_unresolved"))
        if (
            obs.value is not None
            and obs.reasons
            or obs.lexical_class.startswith("invalid")
            or obs.flag == -99
            or obs.quality == "invalid_status"
            or row.contradictions
        ):
            fatal.extend(f"month:{month}:{r}" for r in obs.reasons)
        observations.append(obs)
    fatal.extend(
        f"contradictory_annual_route:{f}"
        for f, values in route_values.items()
        if len(values) > 1
    )
    valid = {o.value for o in observations if o.usable}
    if len(valid) != 1:
        fatal.append(
            "annual_value_conflict" if len(valid) > 1 else "no_usable_annual_value"
        )
    value = next(iter(valid)) if len(valid) == 1 and not fatal else None
    status = (
        "unknown_annual_once"
        if value is None
        else "annual_once_with_missing_repetitions"
        if any(not o.usable for o in observations)
        else "annual_once_agreement"
    )
    return _measurement(
        "sipp_" + field.lower() + "_annual_once",
        field,
        source,
        spec,
        tuple(observations),
        value,
        status,
        fatal,
    )


def _weight(key: PersonKey, row: SippRow, source: SippSource) -> Measurement:
    lexical, value = _number(row.values, "WPFINWGT", 0, 10000000, -999, integral=False)
    reasons = (
        *row.contradictions,
        *(("WPFINWGT:" + lexical,) if value is None else ()),
    )
    obs = Observation(
        key,
        12,
        "WPFINWGT",
        value,
        lexical,
        "published_person_weight",
        None,
        "source_weight",
        _raw(row.values, ("WPFINWGT",)),
        reasons,
    )
    return Measurement(
        "sipp_december_person_weight",
        "person_weight",
        source,
        "December_2022",
        "weight",
        value if not reasons else None,
        "published_weight" if not reasons else "unknown_weight",
        (obs,),
        reasons,
    )


def december_households(
    rows: Sequence[SippRow],
    *,
    source: SippSource,
    complete_membership: bool,
    copied_fields: tuple[str, ...] = (),
    target_year: int = 2022,
) -> tuple[DecemberHousehold, ...]:
    """Group every supplied December member before any target filtering.

    complete_membership=True is a caller assertion, not independent authentication.
    Invalid/duplicate keys refuse the roster. Unknown weights/copies remain explicit
    measurements, whose require_value() refuses for a strict later adapter.
    """
    _check_source(source, target_year)
    if complete_membership is not True:
        raise MeasurementError("COMPLETE_DECEMBER_MEMBERSHIP_REQUIRED")
    if len(set(copied_fields)) != len(copied_fields) or any(
        f not in _HOUSEHOLD for f in copied_fields
    ):
        raise MeasurementError("UNSUPPORTED_OR_DUPLICATE_HOUSEHOLD_FIELD")
    fields = (
        "ERESIDENCEID",
        "WPFINWGT",
        *(f for name in copied_fields for f in (name, _HOUSEHOLD[name].flag)),
    )
    prepared = _prepare(rows, fields)
    groups = {}
    for key, month, row in prepared:
        if month != 12:
            continue
        _, residence = _number(row.values, "ERESIDENCEID", 100001, 400999, -999)
        if residence is None:
            raise MeasurementError("INVALID_DECEMBER_RESIDENCE")
        groups.setdefault((key[0], int(residence)), []).append((key, row))
    result = []
    for household_key, members in sorted(groups.items()):
        weights = tuple(_weight(key, row, source) for key, row in members)
        weight_reasons = tuple(dict.fromkeys(r for w in weights for r in w.reasons))
        mean = None
        if all(w.value is not None for w in weights):
            with localcontext(_ARITHMETIC_CONTEXT):
                mean = sum((w.value for w in weights), Decimal(0)) / len(weights)
        household_weight = Measurement(
            "sipp_december_mean_member_weight",
            "household_weight",
            source,
            "December_2022_mean_all_members",
            "weight",
            mean,
            "mean_member_weight" if mean is not None else "unknown_weight",
            tuple(o for w in weights for o in w.observations),
            weight_reasons,
        )
        measures = []
        for field in copied_fields:
            spec = _HOUSEHOLD[field]
            obs = tuple(_observe(key, 12, row, field, spec) for key, row in members)
            signatures = {(o.value, o.lexical_class, o.flag, o.quality) for o in obs}
            reasons = [r for o in obs for r in o.reasons]
            if len(signatures) > 1:
                reasons.append("household_copy_conflict")
            value = obs[0].value if not reasons else None
            measures.append(
                _measurement(
                    "sipp_" + field.lower(),
                    field,
                    source,
                    spec,
                    obs,
                    value,
                    "household_copy_agreement"
                    if value is not None
                    else "unknown_household_copy",
                    reasons,
                )
            )
        result.append(
            DecemberHousehold(
                household_key,
                tuple(key for key, _ in members),
                household_weight,
                weights,
                tuple(measures),
            )
        )
    return tuple(result)


def select_household_targets(
    households: Sequence[DecemberHousehold], persons: Iterable[PersonKey]
) -> tuple[DecemberHousehold, ...]:
    """Select already measured households without changing membership or weights."""
    keys = [h.key for h in households]
    members = [p for h in households for p in h.members]
    if len(set(keys)) != len(keys) or len(set(members)) != len(members):
        raise MeasurementError("AMBIGUOUS_DECEMBER_ROSTER")
    if len({h.weight.source for h in households}) > 1:
        raise MeasurementError("HOUSEHOLD_SOURCE_MISMATCH")
    targets = set(persons)
    known = set(members)
    if not targets <= known:
        raise MeasurementError("TARGET_PERSON_NOT_IN_DECEMBER_ROSTER")
    return tuple(h for h in households if targets.intersection(h.members))
