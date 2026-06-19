"""UK release-gate helpers.

The UK replacement contract is anchored to the enhanced FRS incumbent: a
Populace artifact must cover the same calibration target surface and must not
silently narrow the exported dataset surface. Candidate-only columns are allowed
only when they are reviewed structural/model-input additions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from populace.build.gates import GateReport, GateResult, export_surface_gate
from populace.build.gates import target_surface_gate as _target_surface_gate

__all__ = [
    "UK_ALLOWED_EXTRA_EXPORT_COLUMNS",
    "UK_CANDIDATE_DATASET_NAME",
    "UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS",
    "UK_MAX_TARGET_ABS_RELATIVE_ERROR",
    "UK_REFERENCE_DATASET_NAME",
    "UK_REVIEWED_EXPORT_EXCLUSIONS",
    "uk_export_surface_gate",
    "uk_release_gate_report",
    "uk_target_fit_gate",
    "uk_target_surface_gate",
]

UK_CANDIDATE_DATASET_NAME = "populace_uk_2023"
UK_REFERENCE_DATASET_NAME = "enhanced_frs_2023_24_recalibrated"
UK_MAX_TARGET_ABS_RELATIVE_ERROR = 0.25

# Reviewed candidate-only fields from the Populace UK prototype. These are
# either clone/source provenance fields or real extra modeled inputs whose
# presence should not make the export-surface gate fail.
UK_ALLOWED_EXTRA_EXPORT_COLUMNS: tuple[str, ...] = (
    "benunit.child_benefit_opts_out",
    "household.bus_fare_spending",
    "household.bus_subsidy_spending",
    "household.clone_index",
    "household.constituency_code_oa",
    "household.consumer_debt",
    "household.electricity_consumption",
    "household.gas_consumption",
    "household.has_fuel_consumption",
    "household.household_is_capital_gains_clone",
    "household.household_is_spi_synthetic",
    "household.la_code_oa",
    "household.lsoa_code",
    "household.mortgage_debt",
    "household.msoa_code",
    "household.num_vehicles",
    "household.oa_code",
    "household.property_purchased",
    "household.rail_usage",
    "household.region_code_oa",
    "person.aa_category",
    "person.age_started_or_accepted_current_education_or_training",
    "person.attends_private_school_random_draw",
    "person.charitable_investment_gifts",
    "person.dla_m_category",
    "person.dla_sc_category",
    "person.esa_health_condition_proxy",
    "person.esa_support_group_proxy",
    "person.employment_sector",
    "person.gift_aid",
    "person.higher_earner_tie_break",
    "person.highest_education",
    "person.is_before_universal_credit_qualifying_young_person_terminal_date",
    "person.is_in_non_advanced_education",
    "person.is_parent",
    "person.legacy_jobseeker_proxy",
    "person.pension_contributions_via_salary_sacrifice",
    "person.pip_dl_category",
    "person.pip_m_category",
    "person.receives_benefits_in_own_right",
    "person.salary_sacrifice_asked",
    "person.salary_sacrifice_reported",
    "person.sic_industry_division",
    "person.student_loan_balance",
    "person.student_loan_plan",
    "person.would_claim_marriage_allowance",
    "person.would_claim_scp",
)

# Reference columns the candidate must still materialize before a swap. These
# are formula-populated, non-zero fields in the incumbent-compatible surface.
UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS: tuple[str, ...] = (
    "person.attends_private_school",
    "person.is_higher_earner",
)

# Reviewed reference-only columns that should not be exported by the candidate.
UK_REVIEWED_EXPORT_EXCLUSIONS: Mapping[str, str] = {
    "person.incapacity_benefit_reported": (
        "The enhanced FRS stores this legacy reported-benefit input as an "
        "all-zero layer; Populace's exported-nonzero gate requires dead zero "
        "layers to be dropped rather than preserved."
    ),
}


def _reviewed_export_exclusions(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    exclusions = dict(UK_REVIEWED_EXPORT_EXCLUSIONS)
    if overrides:
        exclusions.update(overrides)
    hard_exclusions = sorted(
        set(exclusions) & set(UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS)
    )
    if hard_exclusions:
        raise ValueError(
            "UK export-surface reviewed exclusions cannot waive hard-required "
            f"reference columns: {hard_exclusions}."
        )
    return exclusions


def uk_export_surface_gate(
    candidate_columns: Iterable[str],
    reference_columns: Iterable[str],
    *,
    allowed_extra_columns: Iterable[str] = UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Run the UK incumbent-compatibility export-surface gate.

    Missing enhanced-FRS columns fail unless they are reviewed dead/reference-
    only exclusions. Candidate extras default to the reviewed Populace UK
    additions above. A reviewed exclusion is also banned from the candidate
    export surface: dead zero reference layers should be dropped, not shipped.
    """

    candidate = {str(name) for name in candidate_columns}
    exclusions = _reviewed_export_exclusions(reviewed_exclusions)
    result = export_surface_gate(
        candidate,
        reference_columns,
        candidate_name=UK_CANDIDATE_DATASET_NAME,
        reference_name=UK_REFERENCE_DATASET_NAME,
        allowed_extra_columns=allowed_extra_columns,
        reviewed_exclusions=exclusions,
    )
    forbidden_candidate_columns = sorted(candidate & set(exclusions))
    failures = [*result.failures]
    if forbidden_candidate_columns:
        failures.append(
            f"{UK_CANDIDATE_DATASET_NAME}: exports "
            f"{len(forbidden_candidate_columns)} reviewed reference-only "
            f"column(s) that must be dropped: {forbidden_candidate_columns[:20]}."
        )

    return GateResult(
        name=result.name,
        passed=not failures,
        failures=tuple(failures),
        details={
            **dict(result.details),
            "forbidden_candidate_columns": forbidden_candidate_columns,
        },
    )


def uk_target_surface_gate(
    candidate_targets: Iterable[str],
    reference_targets: Iterable[str],
) -> GateResult:
    """Require the UK candidate calibration surface to cover enhanced FRS."""

    return _target_surface_gate(
        candidate_targets,
        reference_targets,
        candidate_name=UK_CANDIDATE_DATASET_NAME,
        reference_name=UK_REFERENCE_DATASET_NAME,
    )


def uk_target_fit_gate(
    target_relative_errors: Mapping[str, float],
    *,
    max_abs_relative_error: float = UK_MAX_TARGET_ABS_RELATIVE_ERROR,
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Fail UK artifacts with severe shipped-weight target errors."""

    exclusions = dict(reviewed_exclusions or {})
    failures_by_target = {
        str(name): float(error)
        for name, error in target_relative_errors.items()
        if abs(float(error)) > max_abs_relative_error and str(name) not in exclusions
    }
    worst_targets = sorted(
        failures_by_target,
        key=lambda name: abs(failures_by_target[name]),
        reverse=True,
    )
    failures = [
        f"{UK_CANDIDATE_DATASET_NAME}: {name} relative error "
        f"{failures_by_target[name]:+.1%} exceeds "
        f"{max_abs_relative_error:.0%}."
        for name in worst_targets[:20]
    ]
    return GateResult(
        name="target_fit",
        passed=not failures,
        failures=tuple(failures),
        details={
            "candidate_name": UK_CANDIDATE_DATASET_NAME,
            "targets_checked": len(target_relative_errors),
            "max_abs_relative_error": max_abs_relative_error,
            "reviewed_exclusions": exclusions,
            "failing_targets": {
                name: failures_by_target[name] for name in worst_targets
            },
        },
    )


def uk_release_gate_report(
    *,
    candidate_columns: Iterable[str],
    reference_columns: Iterable[str],
    candidate_targets: Iterable[str],
    reference_targets: Iterable[str],
    target_relative_errors: Mapping[str, float],
    reviewed_export_exclusions: Mapping[str, str] | None = None,
    reviewed_target_fit_exclusions: Mapping[str, str] | None = None,
) -> GateReport:
    """Combine the UK release gates into one report."""

    return GateReport(
        (
            uk_export_surface_gate(
                candidate_columns,
                reference_columns,
                reviewed_exclusions=reviewed_export_exclusions,
            ),
            uk_target_surface_gate(
                candidate_targets,
                reference_targets,
            ),
            uk_target_fit_gate(
                target_relative_errors,
                reviewed_exclusions=reviewed_target_fit_exclusions,
            ),
        )
    )
