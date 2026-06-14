"""US fiscal target-profile requirements.

These are release-gate requirements for the US target registry, not a scoring
harness. In particular, JCT tax-expenditure rows must be computed from simple
neutralization reforms: run the baseline income tax, neutralize one provision,
run income tax again, and use ``reform_income_tax - baseline_income_tax`` as
the per-household calibration row. That matches the legacy eCPS JCT target
semantics and avoids treating tax expenditures as ordinary aggregate columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from populace.build.gates import TargetCoverageRequirement

__all__ = [
    "US_FISCAL_MACRO_REALISM_BANDS",
    "US_FISCAL_TARGET_COVERAGE_REQUIREMENTS",
    "US_JCT_TAX_EXPENDITURE_REFORMS",
    "SimpleTaxExpenditureReform",
]

TaxExpenditureReformKind = Literal["neutralize_variable"]
TaxExpenditureMatrixRow = Literal["reform_minus_baseline_income_tax"]


@dataclass(frozen=True)
class SimpleTaxExpenditureReform:
    """A JCT target row backed by one simple neutralization reform.

    The builder implementation should construct a PolicyEngine reform that
    calls ``neutralize_variable(neutralized_variable)``, calculate
    ``income_tax`` under both baseline and reform, and calibrate to the
    household-level delta ``reform - baseline``.
    """

    target_name: str
    neutralized_variable: str
    source: str
    kind: TaxExpenditureReformKind = "neutralize_variable"
    output_variable: str = "income_tax"
    matrix_row: TaxExpenditureMatrixRow = "reform_minus_baseline_income_tax"

    def __post_init__(self) -> None:
        if not self.target_name:
            raise ValueError("target_name is required.")
        if not self.neutralized_variable:
            raise ValueError(f"{self.target_name}: neutralized_variable is required.")
        if not self.source:
            raise ValueError(f"{self.target_name}: source is required.")
        if self.kind != "neutralize_variable":
            raise ValueError(
                f"{self.target_name}: JCT targets must use a simple "
                "neutralize_variable reform."
            )
        if self.output_variable != "income_tax":
            raise ValueError(
                f"{self.target_name}: JCT targets must fit the income_tax delta."
            )
        if self.matrix_row != "reform_minus_baseline_income_tax":
            raise ValueError(
                f"{self.target_name}: JCT matrix row must be reform income_tax "
                "minus baseline income_tax."
            )

    def coverage_requirement(self) -> TargetCoverageRequirement:
        """The target-profile requirement satisfied by this JCT row."""
        return TargetCoverageRequirement(
            requirement_id=f"jct_tax_expenditure:{self.neutralized_variable}",
            label=f"JCT tax expenditure for {self.neutralized_variable}",
            accepted_names=(self.target_name,),
            required_metadata=(
                ("kind", self.kind),
                ("output_variable", self.output_variable),
                ("matrix_row", self.matrix_row),
                ("neutralized_variable", self.neutralized_variable),
            ),
            notes=(
                "Must be computed as a simple neutralize_variable reform and "
                "calibrated to income_tax(reform) - income_tax(baseline)."
            ),
        )


US_JCT_TAX_EXPENDITURE_REFORMS: tuple[SimpleTaxExpenditureReform, ...] = (
    SimpleTaxExpenditureReform(
        target_name="nation/jct/salt_deduction_expenditure",
        neutralized_variable="salt_deduction",
        source="Joint Committee on Taxation tax expenditure estimate",
    ),
    SimpleTaxExpenditureReform(
        target_name="nation/jct/medical_expense_deduction_expenditure",
        neutralized_variable="medical_expense_deduction",
        source="Joint Committee on Taxation tax expenditure estimate",
    ),
    SimpleTaxExpenditureReform(
        target_name="nation/jct/charitable_deduction_expenditure",
        neutralized_variable="charitable_deduction",
        source="Joint Committee on Taxation tax expenditure estimate",
    ),
    SimpleTaxExpenditureReform(
        target_name="nation/jct/interest_deduction_expenditure",
        neutralized_variable="interest_deduction",
        source="Joint Committee on Taxation tax expenditure estimate",
    ),
    SimpleTaxExpenditureReform(
        target_name="nation/jct/qualified_business_income_deduction_expenditure",
        neutralized_variable="qualified_business_income_deduction",
        source="Joint Committee on Taxation tax expenditure estimate",
    ),
)

US_FISCAL_TARGET_COVERAGE_REQUIREMENTS: tuple[TargetCoverageRequirement, ...] = (
    TargetCoverageRequirement(
        requirement_id="federal_income_tax_total",
        label="Federal individual income tax total",
        accepted_names=(
            "nation/treasury/individual_income_tax",
            "nation/treasury/individual income tax",
            "nation/irs/income_tax_total",
            "nation/irs/total_income_tax",
            "nation/irs/total income tax",
        ),
        accepted_name_prefixes=(
            "nation/treasury/individual_income_tax/",
            "nation/treasury/individual income tax/",
            "nation/irs/income_tax_total/",
            "nation/irs/total_income_tax/",
            "nation/irs/total income tax/",
        ),
        notes=(
            "A positive-income-tax diagnostic is not enough; the profile needs "
            "the total federal income tax aggregate used by downstream netting."
        ),
    ),
    TargetCoverageRequirement(
        requirement_id="irs_agi_distribution",
        label="SOI AGI distribution and top-tail controls",
        accepted_name_substrings=("/irs/adjusted gross income/",),
        min_matches=20,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_wages_distribution",
        label="SOI wages by AGI bracket",
        accepted_name_substrings=("/irs/salaries and wages/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_business_income_distribution",
        label="SOI business income by AGI bracket",
        accepted_name_substrings=("/irs/business net ",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_partnership_s_corp_distribution",
        label="SOI partnership and S-corp income by AGI bracket",
        accepted_name_substrings=("/irs/partnership and s corp income/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_capital_gains_distribution",
        label="SOI capital gains by AGI bracket",
        accepted_name_substrings=("/irs/capital gains gross/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_dividends_distribution",
        label="SOI dividends by AGI bracket",
        accepted_name_substrings=("/irs/ordinary dividends/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_interest_distribution",
        label="SOI taxable interest by AGI bracket",
        accepted_name_substrings=("/irs/taxable interest income/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_pension_distribution",
        label="SOI pension income by AGI bracket",
        accepted_name_substrings=("/irs/total pension income/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="irs_social_security_distribution",
        label="SOI Social Security by AGI bracket",
        accepted_name_substrings=("/irs/total social security/",),
        min_matches=5,
    ),
    TargetCoverageRequirement(
        requirement_id="state_income_tax",
        label="State individual income tax collections",
        accepted_names=("state_income_tax",),
        accepted_name_substrings=(
            "/state_income_tax",
            "/state income tax",
            "/individual_income_tax",
            "/state individual income tax",
        ),
        accepted_measures=("state_income_tax",),
        min_matches=50,
    ),
    *(spec.coverage_requirement() for spec in US_JCT_TAX_EXPENDITURE_REFORMS),
)

US_FISCAL_MACRO_REALISM_BANDS: dict[str, tuple[float, float]] = {
    "federal_income_tax_to_gdp": (0.07, 0.11),
    "agi_to_gdp": (0.50, 0.70),
    "spm_poverty_rate": (0.06, 0.18),
}
