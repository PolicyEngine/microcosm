"""Shared US critical-target requirements for building and publishing.

The US release builder and the publication contract are two independent gates
over the same calibration diagnostics. Keeping their requirement values in
one dependency-light module makes it impossible for a release build to pass a
weaker hand-copied register than the publisher later enforces (populace#462).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR",
    "US_CRITICAL_DEDUCTION_MAX_ABS_RELATIVE_ERROR",
    "US_CRITICAL_TARGET_FIT_REQUIREMENTS",
    "US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR",
    "US_EXACT_CRITICAL_TARGET_FIT_REQUIREMENTS",
    "US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT",
    "USCriticalTargetFitRequirement",
    "is_congressional_district_target",
]


@dataclass(frozen=True)
class USCriticalTargetFitRequirement:
    """One shared critical-target row class and its publication tolerance."""

    requirement_id: str
    label: str
    max_abs_relative_error: float
    names: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    target_roles: tuple[str, ...] = ()
    name_substrings: tuple[str, ...] = ()
    name_suffixes: tuple[str, ...] = ()
    allow_incumbent_improvement: bool = True

    def matches(self, *, name: str, family: str = "", target_role: str = "") -> bool:
        """Return whether a diagnostic row belongs to this requirement.

        Exact names, name patterns, and the family/role pair are alternative
        selectors. Pattern fields are conjunctive across fields and
        disjunctive within a field.
        """
        exact_match = name in self.names
        has_name_pattern = bool(self.name_substrings or self.name_suffixes)
        pattern_match = (
            has_name_pattern
            and (
                not self.name_substrings
                or any(part in name for part in self.name_substrings)
            )
            and (
                not self.name_suffixes
                or any(name.endswith(suffix) for suffix in self.name_suffixes)
            )
        )
        semantic_match = (
            bool(self.families and self.target_roles)
            and family in self.families
            and target_role in self.target_roles
        )
        return exact_match or pattern_match or semantic_match


def is_congressional_district_target(
    name: object,
    metadata: Mapping | None,
) -> bool:
    """Return whether any shared evidence identifies a US CD target."""
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return (
        str(metadata.get("ledger_layout_groupby_dimension") or "")
        == "irs_soi.congressional_district"
        or ".congressional_district_"
        in str(metadata.get("ledger_source_record_id") or "")
        or str(metadata.get("ledger_geography_level") or "") == "congressional_district"
        or str(metadata.get("geography_scope") or "") == "congressional_district"
        or bool(metadata.get("congressional_district_geoid"))
        or ".congressional_district_" in str(name or "")
    )


US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR = 0.15
US_CRITICAL_DEDUCTION_MAX_ABS_RELATIVE_ERROR = 0.15
US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR = 0.25

US_EXACT_CRITICAL_TARGET_FIT_REQUIREMENTS = (
    USCriticalTargetFitRequirement(
        requirement_id="federal_income_tax_amount",
        label="federal income tax liability amount",
        max_abs_relative_error=0.05,
        names=(
            "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount@2024",
        ),
        families=("irs_soi",),
        target_roles=("federal_income_tax_total",),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="income_tax_liability_returns",
        label="income tax liability returns",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=(
            "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_returns@2024",
        ),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="social_security_benefits",
        label="Social Security benefits",
        max_abs_relative_error=0.05,
        names=(
            "ssa_supplement.cy2024.oasdi_ssi_payments."
            "social_security_benefits.payment_amount@2024",
        ),
        families=("ssa",),
        target_roles=("social_security_total",),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="ctc_amount",
        label="Child Tax Credit amount",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=("irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024",),
        families=("irs_soi",),
        target_roles=("ctc_total",),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="ctc_claims",
        label="Child Tax Credit claims",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=("irs_soi.ty2022.historic_table_2.us.all.ctc_claims@2024",),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="actc_amount",
        label="Additional Child Tax Credit amount",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=("irs_soi.ty2022.historic_table_2.us.all.actc_amount@2024",),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="actc_claims",
        label="Additional Child Tax Credit claims",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=("irs_soi.ty2022.historic_table_2.us.all.actc_claims@2024",),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="eitc_amount",
        label="Earned Income Tax Credit amount",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=(
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount@2024",
        ),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="eitc_claims",
        label="Earned Income Tax Credit claims",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=(
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_returns@2024",
        ),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="premium_tax_credit_amount",
        label="Premium Tax Credit amount",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=(
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount@2024",
        ),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="premium_tax_credit_returns",
        label="Premium Tax Credit returns",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=(
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns@2024",
        ),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="taxable_social_security_amount",
        label="taxable Social Security amount",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=(
            "irs_soi.ty2022.historic_table_2.us.all."
            "taxable_social_security_amount@2024",
        ),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="taxable_social_security_returns",
        label="taxable Social Security returns",
        max_abs_relative_error=US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
        names=(
            "irs_soi.ty2022.historic_table_2.us.all."
            "taxable_social_security_returns@2024",
        ),
    ),
    USCriticalTargetFitRequirement(
        requirement_id="itemized_deduction_amount",
        label="itemized deduction amount",
        max_abs_relative_error=US_CRITICAL_DEDUCTION_MAX_ABS_RELATIVE_ERROR,
        names=(
            "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount@2024",
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "total_itemized_deductions_amount@2024",
        ),
        families=("irs_soi",),
        target_roles=("itemized_deduction_total",),
        allow_incumbent_improvement=False,
    ),
    USCriticalTargetFitRequirement(
        requirement_id="salt_deduction_amount",
        label="state and local tax deduction amount",
        max_abs_relative_error=0.10,
        names=(
            "irs_soi.ty2022.historic_table_2.us.all."
            "limited_state_local_taxes_amount@2024",
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "limited_state_local_taxes_amount@2024",
        ),
        families=("irs_soi",),
        target_roles=("salt_deduction_total",),
        allow_incumbent_improvement=False,
    ),
    USCriticalTargetFitRequirement(
        requirement_id="medical_expense_deduction_amount",
        label="medical expense deduction amount",
        # 2026-07-22 adjudication (Max): the N release does not hold on this
        # row. Build N's truthful capital-gains/interest target moves
        # (populace#462, #488) re-equilibrated the solve and pushed medical
        # from certified M's +4.4% to +20.8%, stable across a 2x-epoch run.
        # Post-publish decomposition (populace#462, 2026-07-22): the national
        # overshoot ($18.68B) is the six states' excess ($18.80B, within
        # 0.7%) — AR/DC/NV/NY/TN/UT; returns fit everywhere except UT
        # (+146%), amounts blow out; all six amount rows sit past the loss
        # cap. A support defect, not a loss-priority problem, and the
        # owner doctrine (populace#492) forbids per-target loss knobs.
        # Relaxed to the established 0.25 broad-fit bound; restore
        # US_CRITICAL_DEDUCTION_MAX_ABS_RELATIVE_ERROR once the six-state
        # carriers are fixed (owners populace#481/#487) and a run holds 0.15
        # on truthful support.
        max_abs_relative_error=0.25,
        names=(
            "irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_amount@2024",
        ),
        families=("irs_soi",),
        target_roles=("medical_expense_deduction_total",),
        allow_incumbent_improvement=False,
    ),
)

# populace#462: every national SOI Pub 1304 Table 1.4 dollar row is
# within-tolerance-blocking by name pattern, not enumeration. The Build M live
# default shipped capital-gain-distributions at +634.8% because no exact-name
# entry covered it. There is no incumbent-improvement escape for this blanket.
US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT = USCriticalTargetFitRequirement(
    requirement_id="soi_table_1_4_national_dollar_rows",
    label="SOI Pub 1304 Table 1.4 national dollar rows",
    max_abs_relative_error=0.25,
    name_substrings=(".table_1_4.",),
    name_suffixes=("_amount@2024",),
    allow_incumbent_improvement=False,
)

US_CRITICAL_TARGET_FIT_REQUIREMENTS = (
    *US_EXACT_CRITICAL_TARGET_FIT_REQUIREMENTS,
    US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT,
)
