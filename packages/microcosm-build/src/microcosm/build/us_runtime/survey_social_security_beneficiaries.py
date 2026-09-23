"""Pure, development-only report-lot and beneficiary allocation contracts.

Public dataclasses describe claims; constructing or validating one does not
authenticate a survey source, a complete roster, or a beneficiary observation.
The only implemented assignment is an explicitly accepted singleton ASEC
retirement convention. Unknown amounts and known report zeros remain report
evidence. No report classifier probabilities become beneficiary incidence.

A later source-owning graph adapter must bind original identities, physical
source seals, roster evidence, period/definition and exact clone transport.
This module performs no reads, fits, engine calls, Frame writes or releases.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .survey_social_security import COMPONENTS

PROTOCOL = "microcosm.us.social-security-beneficiary-convention.v1"
ASEC_OASDI_DEFINITION = "asec_social_security_oasdi"
SINGLETON_CONVENTION = "development_asec_singleton_retirement_v1"
SINGLETON_ASSUMPTIONS = (
    "closed_benefit_unit",
    "stable_roster_over_income_year",
)


def _require(condition, reason):
    if not condition:
        raise ValueError("SS_BENEFICIARY_" + reason)


def _text(value):
    return type(value) is str and bool(value.strip())


@dataclass(frozen=True)
class ReportLot:
    """One original source-person report, in USD; ``None`` is unknown money.

    The source coordinates are descriptive until a source owner binds them.
    Amount/period edits do not create a new report identity. ``reporter_age``
    locates the reporting universe, not statutory beneficiary eligibility.
    """

    survey: str
    source_vintage: str
    reporter_id: str
    amount: float | None
    receipt_code: int | None
    reporter_age: int | None
    reason_codes: tuple[int | None, int | None]
    publisher_allocation: str
    definition: str
    period_basis: str
    income_year: int | None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.survey, self.source_vintage, self.reporter_id


@dataclass(frozen=True)
class RosterEvidence:
    """Descriptive interview roster; not a source-issued completeness proof."""

    person_ids: tuple[str, ...]
    complete: bool | None
    housing_unit: bool | None


@dataclass(frozen=True)
class BeneficiaryCandidate:
    """Possible edge without dollars; an unknown/nonresident person may be None."""

    beneficiary_id: str | None
    components: tuple[str, ...]
    relationship_basis: str


@dataclass(frozen=True)
class BeneficiaryAssignment:
    beneficiary_id: str
    component: str
    amount: float


@dataclass(frozen=True)
class LotAllocation:
    report: ReportLot
    candidates: tuple[BeneficiaryCandidate, ...]
    assignments: tuple[BeneficiaryAssignment, ...]
    assignment_resolution: str
    unresolved_amount: float | None
    definition_resolution: str
    period_resolution: str
    convention: str | None
    accepted_assumptions: tuple[str, ...]
    roster: RosterEvidence | None


@dataclass(frozen=True)
class BeneficiaryInput:
    """Annual USD four-vector ordered by the existing report COMPONENTS tuple."""

    beneficiary_id: str
    values: tuple[float, float, float, float]
    value_origin: str


@dataclass(frozen=True)
class DevelopmentConsumerSubset:
    """Structurally ready convention subset, never an admitted engine input.

    The caller must separately qualify actual source bindings and consumer
    compatibility. This object claims no completeness beyond its exact rows.
    Its construction remains public and conveys no capability or authority.
    """

    rows: tuple[BeneficiaryInput, ...]
    report_keys: tuple[tuple[str, str, str], ...]
    income_year: int
    definition: str

    @property
    def development_only(self):
        return True

    @property
    def source_admission_issued(self):
        return False

    @property
    def observed_beneficiary_income_claim(self):
        return False

    @property
    def engine_compatibility_qualified(self):
        return False

    @property
    def release_eligible(self):
        return False


def _validate_report(report):
    _require(type(report) is ReportLot, "REPORT_TYPE")
    _require(report.survey in ("asec", "acs"), "SURVEY")
    _require(_text(report.source_vintage) and _text(report.reporter_id), "REPORT_KEY")
    _require(
        report.amount is None
        or (
            type(report.amount) is float
            and isfinite(report.amount)
            and report.amount >= 0
        ),
        "REPORT_AMOUNT",
    )
    _require(
        report.receipt_code is None
        or (type(report.receipt_code) is int and report.receipt_code in (0, 1, 2)),
        "RECEIPT_CODE",
    )
    _require(
        report.reporter_age is None
        or (type(report.reporter_age) is int and report.reporter_age >= 0),
        "REPORT_AGE",
    )
    _require(
        type(report.reason_codes) is tuple
        and len(report.reason_codes) == 2
        and all(
            code is None or (type(code) is int and 0 <= code <= 8)
            for code in report.reason_codes
        ),
        "REASONS",
    )
    _require(
        report.publisher_allocation
        in ("publisher_no_allocation", "publisher_allocated", "unresolved"),
        "ALLOCATION_STATUS",
    )
    _require(_text(report.definition), "DEFINITION_LABEL")
    _require(
        report.period_basis in ("calendar_year", "rolling_12_months", "unknown"),
        "PERIOD_BASIS",
    )
    _require(
        report.income_year is None
        or (type(report.income_year) is int and report.income_year > 0),
        "INCOME_YEAR",
    )


def _validate_candidates(candidates):
    _require(type(candidates) is tuple, "CANDIDATES")
    for candidate in candidates:
        _require(type(candidate) is BeneficiaryCandidate, "CANDIDATE_TYPE")
        _require(
            candidate.beneficiary_id is None or _text(candidate.beneficiary_id),
            "CANDIDATE_ID",
        )
        _require(
            type(candidate.components) is tuple
            and bool(candidate.components)
            and all(
                type(component) is str and component in COMPONENTS
                for component in candidate.components
            )
            and len(set(candidate.components)) == len(candidate.components),
            "CANDIDATE_COMPONENTS",
        )
        _require(_text(candidate.relationship_basis), "CANDIDATE_BASIS")


def _check_singleton(report, roster, assumptions):
    _validate_report(report)
    _require(
        type(assumptions) is tuple and assumptions == SINGLETON_ASSUMPTIONS,
        "ASSUMPTIONS",
    )
    _require(report.survey == "asec", "ASEC_ONLY")
    _require(report.amount is not None and report.amount > 0, "POSITIVE_REPORT")
    _require(report.receipt_code == 1, "REPORT_RECEIPT")
    _require(
        report.reporter_age is not None and report.reporter_age >= 15, "REPORT_UNIVERSE"
    )
    _require(
        all(code in (0, 1) for code in report.reason_codes)
        and 1 in report.reason_codes,
        "RETIREMENT_ONLY",
    )
    _require(
        report.publisher_allocation == "publisher_no_allocation", "ALLOCATION_ORIGIN"
    )
    _require(report.definition == ASEC_OASDI_DEFINITION, "DEFINITION")
    _require(
        report.period_basis == "calendar_year" and report.income_year is not None,
        "CALENDAR_YEAR",
    )
    _require(type(roster) is RosterEvidence, "ROSTER_TYPE")
    _require(
        type(roster.person_ids) is tuple
        and roster.person_ids == (report.reporter_id,)
        and roster.complete is True
        and roster.housing_unit is True,
        "SINGLETON_ROSTER",
    )


def _singleton_candidate(report):
    return BeneficiaryCandidate(
        report.reporter_id,
        (COMPONENTS[0],),
        "singleton_roster_with_accepted_closure_assumptions",
    )


def unresolved_report_lot(report: ReportLot, *, candidates=()) -> LotAllocation:
    """Preserve all report money as unresolved, including known report zero.

    Candidates are descriptive possibilities. Neither candidate counts, class
    scores nor a reporting-age exclusion identify shares or beneficiary zeros.
    """
    _validate_report(report)
    _validate_candidates(candidates)
    return LotAllocation(
        report,
        candidates,
        (),
        "unresolved",
        report.amount,
        "unresolved",
        "unresolved",
        None,
        (),
        None,
    )


def resolve_singleton_asec_retirement(
    report: ReportLot,
    roster: RosterEvidence,
    *,
    closed_benefit_unit: bool,
    stable_roster_over_income_year: bool,
) -> LotAllocation:
    """Opt into two assumptions for one positive, retirement-only ASEC report.

    Closure means no nonresident benefit in this report and no benefit of this
    resident reported elsewhere. Stability means the interview roster is an
    adequate boundary throughout the income year. Both are assumptions, not
    observations. No statutory age eligibility or benefits calculation occurs.
    """
    _require(
        closed_benefit_unit is True and stable_roster_over_income_year is True,
        "ASSUMPTIONS",
    )
    _check_singleton(report, roster, SINGLETON_ASSUMPTIONS)
    return LotAllocation(
        report,
        (_singleton_candidate(report),),
        (BeneficiaryAssignment(report.reporter_id, COMPONENTS[0], report.amount),),
        "resolved_by_convention",
        0.0,
        "asec_oasdi_by_convention",
        "source_calendar_year",
        SINGLETON_CONVENTION,
        SINGLETON_ASSUMPTIONS,
        roster,
    )


def validate_allocation(allocation: LotAllocation) -> None:
    """Revalidate descriptive content; this does not requalify source evidence."""
    _require(type(allocation) is LotAllocation, "ALLOCATION_TYPE")
    _validate_report(allocation.report)
    _validate_candidates(allocation.candidates)
    _require(type(allocation.assignments) is tuple, "ASSIGNMENTS")
    _require(
        allocation.assignment_resolution in ("unresolved", "resolved_by_convention"),
        "RESOLUTION",
    )
    if allocation.assignment_resolution == "unresolved":
        _require(not allocation.assignments, "UNRESOLVED_ASSIGNMENTS")
        _require(
            type(allocation.unresolved_amount) is type(allocation.report.amount)
            and allocation.unresolved_amount == allocation.report.amount,
            "UNRESOLVED_AMOUNT",
        )
        _require(
            allocation.definition_resolution == "unresolved", "DEFINITION_RESOLUTION"
        )
        _require(allocation.period_resolution == "unresolved", "PERIOD_RESOLUTION")
        _require(
            allocation.convention is None
            and allocation.accepted_assumptions == ()
            and allocation.roster is None,
            "UNRESOLVED_CONVENTION",
        )
        return
    _require(allocation.convention == SINGLETON_CONVENTION, "CONVENTION")
    _check_singleton(
        allocation.report, allocation.roster, allocation.accepted_assumptions
    )
    _require(
        allocation.definition_resolution == "asec_oasdi_by_convention",
        "DEFINITION_RESOLUTION",
    )
    _require(
        allocation.period_resolution == "source_calendar_year", "PERIOD_RESOLUTION"
    )
    _require(
        type(allocation.unresolved_amount) is float
        and allocation.unresolved_amount == 0.0,
        "UNRESOLVED_AMOUNT",
    )
    _require(
        allocation.candidates == (_singleton_candidate(allocation.report),),
        "CANDIDATE_CONVENTION",
    )
    # A single exact edge conserves the full lot without floating tolerances,
    # rounding, equal splits or normalization of positive category scores.
    expected = BeneficiaryAssignment(
        allocation.report.reporter_id, COMPONENTS[0], allocation.report.amount
    )
    _require(
        len(allocation.assignments) == 1
        and type(allocation.assignments[0]) is BeneficiaryAssignment
        and type(allocation.assignments[0].amount) is float
        and allocation.assignments[0] == expected,
        "ASSIGNMENT_CONVENTION",
    )


def validate_development_consumer_subset(
    allocations: tuple[LotAllocation, ...],
    *,
    beneficiary_ids: tuple[str, ...],
    income_year: int,
) -> DevelopmentConsumerSubset:
    """Return only an exact, resolved development subset; fail on unknowns.

    Every supplied lot must resolve. Repeated original coordinates are refused
    even when their payloads differ. Distinct reports for one beneficiary are
    also refused: this slice has no overlap/deduplication model. The three
    nonretirement zeros follow the accepted convention, never report-zero or
    unknown-zero fallback. No clone or engine entity binding is performed.
    """
    _require(type(allocations) is tuple and bool(allocations), "SUBSET_ALLOCATIONS")
    _require(
        type(beneficiary_ids) is tuple
        and bool(beneficiary_ids)
        and all(_text(person) for person in beneficiary_ids)
        and len(set(beneficiary_ids)) == len(beneficiary_ids),
        "SUBSET_IDS",
    )
    _require(type(income_year) is int and income_year > 0, "CONSUMER_PERIOD")
    keys, by_person = set(), {}
    for allocation in allocations:
        validate_allocation(allocation)
        report = allocation.report
        _require(report.key not in keys, "DUPLICATE_REPORT_LOT")
        keys.add(report.key)
        _require(
            allocation.assignment_resolution == "resolved_by_convention",
            "UNRESOLVED_ASSIGNMENT",
        )
        _require(report.income_year == income_year, "CONSUMER_PERIOD")
        _require(report.reporter_id not in by_person, "OVERLAPPING_BENEFICIARY_LOTS")
        by_person[report.reporter_id] = allocation
    _require(set(by_person) == set(beneficiary_ids), "SUBSET_COVERAGE")
    return DevelopmentConsumerSubset(
        rows=tuple(
            BeneficiaryInput(
                person,
                (by_person[person].report.amount, 0.0, 0.0, 0.0),
                "singleton_retirement_convention",
            )
            for person in beneficiary_ids
        ),
        report_keys=tuple(by_person[person].report.key for person in beneficiary_ids),
        income_year=income_year,
        definition=ASEC_OASDI_DEFINITION,
    )
