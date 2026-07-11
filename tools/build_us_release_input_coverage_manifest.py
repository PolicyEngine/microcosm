"""Regenerate the US release input-column coverage manifest (populace #368).

The manifest is the declared column-coverage contract the release gate enforces:
every input column the reference eCPS exports must be persisted by a populace
release as a real key with non-default signal, or carry a reviewed exclusion.

Derivation (fully from checked-in, sha-pinned facts — no transient artifact):

- Required surface = every input-variable column the pinned reference eCPS
  populates, i.e. the ``nonzero_shares`` keys of ``ecps_parity_reference.json``,
  plus explicit later inputs needed by shipped reform probes
  (computed once from the sha-verified ``enhanced_cps_2024.h5``; an input the
  incumbent exports but leaves all-zero is not a coverage requirement, the same
  rule the parity gate uses).
- Status per column:
    * ``reviewed_exclusion`` — the column is a documented incumbent-parity gap
      (an entry in ``ecps_parity_known_gaps.json``, carrying that register's
      reason and tracking issue), so the current candidate does not populate it
      yet. EXCEPT the SSI countable-resource asset inputs (below).
    * ``required`` — every other populated layer, PLUS the SSI countable-resource
      asset inputs. Per #368 ("it must be an actual gate"), the asset inputs get
      NO exclusion even though they are currently absent, so the gate ships red
      for today's artifacts and Deliverable 2 (asset restoration) turns it green.

The SSI asset inputs are ``bank_account_assets``, ``stock_assets``, and
``bond_assets`` — the exact ``adds`` list of ``ssi_countable_resources``. With
them absent, countable resources are 0 for every record, so the SSI asset-limit
reform probe scores $0; that is the failure the gate exists to surface.

Run:  uv run python tools/build_us_release_input_coverage_manifest.py
It rewrites packages/populace-build/src/populace/build/us/
release_input_coverage_manifest.json. A test asserts the committed file matches
this regeneration, so the manifest cannot silently drift from the pinned eCPS
surface or the parity register.
"""

from __future__ import annotations

import json
from pathlib import Path

US_PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "populace-build"
    / "src"
    / "populace"
    / "build"
    / "us"
)
MANIFEST_PATH = US_PACKAGE_DIR / "release_input_coverage_manifest.json"

#: The ``adds`` list of PolicyEngine-US ``ssi_countable_resources``: the asset
#: leaves SSI eligibility keys on. Absent from the release, they zero countable
#: resources and the SSI asset-limit reform scores $0. Per #368 these ship as
#: hard requirements with NO reviewed exclusion.
SSI_COUNTABLE_RESOURCE_ASSETS = (
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
)

# The pinned reference H5 predates the retired pipeline's FLSA-premium export,
# OBBBA's distinct qualifying passenger-vehicle interest leaf, and its five
# final desired retirement-contribution inputs. These later inputs are hard
# requirements because the shipped validation provisions must bind.
POST_REFERENCE_ECPS_REQUIRED_INPUTS = (
    "fsla_overtime_premium",
    "qualified_passenger_vehicle_loan_interest",
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
)

QBI_INPUTS = (
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
    "self_employment_income_would_be_qualified",
    "sstb_self_employment_income_would_be_qualified",
    "business_is_sstb",
    "qualified_bdc_income",
    "qualified_reit_and_ptp_income",
    "sstb_self_employment_income_before_lsr",
    "sstb_unadjusted_basis_qualified_property",
    "sstb_w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "w2_wages_from_qualified_business",
)

CHILD_SUPPORT_INPUTS = (
    "child_support_received",
    "child_support_expense",
)

DISABILITY_BENEFITS_INPUTS = ("disability_benefits",)

EDUCATOR_EXPENSE_INPUTS = ("educator_expense",)

OTHER_HEALTH_INSURANCE_INPUTS = ("other_health_insurance_premiums",)

FARM_BUSINESS_INCOME_INPUTS = (
    "farm_operations_income",
    "farm_rent_income",
)

SIPP_VEHICLE_INPUTS = (
    "household_vehicles_owned",
    "household_vehicles_value",
)

FORM_4952_INPUTS = ("investment_income_elected_form_4952",)

# Reference-populated inputs whose primary-source restoration has shipped.
# They remain hard requirements even if a stale parity-gap entry is
# accidentally reintroduced later.
RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS = (
    "alimony_expense",
    "alimony_income",
    "casualty_loss",
    *CHILD_SUPPORT_INPUTS,
    *DISABILITY_BENEFITS_INPUTS,
    *EDUCATOR_EXPENSE_INPUTS,
    *OTHER_HEALTH_INSURANCE_INPUTS,
    *FARM_BUSINESS_INCOME_INPUTS,
    *SIPP_VEHICLE_INPUTS,
    *FORM_4952_INPUTS,
    "domestic_production_ald",
    "household_weight",
    "spm_unit_pre_subsidy_childcare_expenses",
    "unreimbursed_business_employee_expenses",
    *QBI_INPUTS,
)

RETIREMENT_CONTRIBUTION_INPUTS = (
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
)

AOTC_EDUCATION_INPUTS = (
    "qualified_tuition_expenses",
    "is_pursuing_credential_for_american_opportunity_credit",
    "attends_eligible_educational_institution_for_american_opportunity_credit",
    "is_enrolled_at_least_half_time_for_american_opportunity_credit",
    "has_american_opportunity_credit_1098_t_or_exception",
    "has_american_opportunity_credit_institution_ein",
)

#: Pinned reform-coverage probes. Raising the SSI resource limit from
#: the 2024 statutory $2,000 individual / $3,000 couple to $10,000 / $20,000 is
#: a pure relaxation that binds only through ``ssi_countable_resources``. Nonzero
#: iff the asset inputs are restored. Dense-native reference magnitudes: +$1.6B
#: at $10k/$20k and ~+$16.1B with no limit (populace #356). ``min_abs_effect`` is
#: a floor far below the plausible effect but far above simulation noise, so a
#: structural $0 fails while a real (even conservative) score passes.
REFORM_COVERAGE_PROBES = [
    {
        "id": "ssi_asset_limit_10k_20k",
        "name": "SSI asset limits raised to $10k individual / $20k couple",
        "parameter_changes": {
            "gov.ssa.ssi.eligibility.resources.limit.individual": {
                "2024-01-01.2100-12-31": 10_000
            },
            "gov.ssa.ssi.eligibility.resources.limit.couple": {
                "2024-01-01.2100-12-31": 20_000
            },
        },
        "budget_measure": "ssi",
        "period": 2024,
        "effect_direction": "reform_minus_baseline",
        "expected_sign": "positive",
        "binding_inputs": list(SSI_COUNTABLE_RESOURCE_ASSETS),
        "min_abs_effect": 1_000_000_000.0,
        "reason": (
            "Raising the SSI resource limit is a pure relaxation that only "
            "changes who passes meets_ssi_resource_test, which compares "
            "ssi_countable_resources = bank_account_assets + stock_assets + "
            "bond_assets against the limit. With those asset inputs absent, "
            "countable resources are 0 for every record, everyone already "
            "passes the resource test, and raising the limit scores exactly "
            "$0. Dense-native reference: +$1.6B at $10k/$20k, ~+$16.1B with no "
            "limit (PolicyEngine/populace#356)."
        ),
        "issue": "PolicyEngine/populace#356",
    },
    {
        "id": "aotc_abolition",
        "name": "American Opportunity Tax Credit abolition",
        "parameter_changes": {
            "gov.irs.credits.education.american_opportunity_credit.abolition": {
                "2024-01-01.2100-12-31": True
            }
        },
        "budget_measure": "american_opportunity_credit",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": list(AOTC_EDUCATION_INPUTS),
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Abolishing the American Opportunity Tax Credit sets the credit "
            "to zero, so baseline-minus-reform AOTC must be positive. With "
            "qualified tuition or any of the five affirmative AOTC factual "
            "inputs absent or degenerate, the baseline credit is a structural "
            "zero and the abolition scores exactly $0."
        ),
        "issue": "PolicyEngine/populace#253",
    },
    {
        "id": "savers_credit_abolition",
        "name": "Retirement Saver's Credit abolition",
        "parameter_changes": {
            "gov.irs.credits.retirement_saving.contributions_cap": {
                "2024-01-01.2100-12-31": 0
            }
        },
        "budget_measure": "savers_credit",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": list(RETIREMENT_CONTRIBUTION_INPUTS),
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting the Saver's Credit contribution cap to zero abolishes "
            "the credit, so baseline-minus-reform Saver's Credit must be "
            "positive. PolicyEngine-US builds qualified contributions from "
            "the realized forms of all five desired retirement-contribution "
            "inputs. If the desired-input family is absent or degenerate, "
            "the baseline credit is a structural zero and abolition scores $0."
        ),
        "issue": "PolicyEngine/populace#278",
    },
    {
        "id": "qbi_reit_ptp_rate_abolition",
        "name": "Section 199A qualified REIT/PTP component abolition",
        "parameter_changes": {
            "gov.irs.deductions.qbi.max.reit_ptp_rate": {"2024-01-01.2100-12-31": 0}
        },
        "budget_measure": "qualified_business_income_deduction",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["qualified_reit_and_ptp_income"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Setting only the qualified REIT/PTP component rate to zero "
            "removes that component from the Section 199A deduction, so "
            "baseline-minus-reform QBID must be positive. Without populated "
            "qualified_reit_and_ptp_income the change is a structural zero."
        ),
        "issue": "PolicyEngine/populace#298",
    },
    {
        "id": "qbi_wage_property_guardrails_zeroed",
        "name": "Section 199A W-2 wage and UBIA guardrails zeroed",
        "parameter_changes": {
            "gov.irs.deductions.qbi.max.w2_wages.rate": {"2024-01-01.2100-12-31": 0},
            "gov.irs.deductions.qbi.max.w2_wages.alt_rate": {
                "2024-01-01.2100-12-31": 0
            },
            "gov.irs.deductions.qbi.max.business_property.rate": {
                "2024-01-01.2100-12-31": 0
            },
        },
        "budget_measure": "qualified_business_income_deduction",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": [
            "w2_wages_from_qualified_business",
            "unadjusted_basis_qualified_property",
        ],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Zeroing all W-2 wage and UBIA cap rates tightens the Section "
            "199A deduction for high-income qualified businesses, so "
            "baseline-minus-reform QBID must be positive. If the total W-2 "
            "and UBIA inputs are absent, both baseline and reform guardrails "
            "are zero and the change is a structural zero. The archived "
            "all-or-nothing SSTB routing leaves its SSTB-allocable copies to "
            "the hard signal gate rather than overclaiming reform coverage."
        ),
        "issue": "PolicyEngine/populace#298",
    },
    {
        "id": "qbi_farm_operations_income_exclusion",
        "name": "Exclude farm-operations income from Section 199A QBI",
        "parameter_changes": {
            "gov.irs.deductions.qbi.income_definition": {
                "2026-01-01.2026-12-31": [
                    "self_employment_income",
                    "partnership_s_corp_income",
                    "farm_rent_income",
                    "rental_income",
                    "estate_income",
                ]
            }
        },
        "budget_measure": "qualified_business_income_deduction",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["farm_operations_income"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only farm_operations_income from the 2026 Section 199A "
            "income definition isolates the restored signed Schedule F leaf. "
            "The real staged candidate is loss-heavy, so excluding it raises "
            "QBID and baseline-minus-reform is negative (-$4.16M). Without "
            "farm_operations_income the reform is a structural zero."
        ),
        "issue": "PolicyEngine/populace#298",
    },
    {
        "id": "qbi_farm_rent_income_exclusion",
        "name": "Exclude farm-rent income from Section 199A QBI",
        "parameter_changes": {
            "gov.irs.deductions.qbi.income_definition": {
                "2026-01-01.2026-12-31": [
                    "self_employment_income",
                    "partnership_s_corp_income",
                    "farm_operations_income",
                    "rental_income",
                    "estate_income",
                ]
            }
        },
        "budget_measure": "qualified_business_income_deduction",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["farm_rent_income"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only farm_rent_income from the 2026 Section 199A income "
            "definition isolates the restored signed E27200 leaf. The real "
            "staged candidate produces +$9.14M baseline-minus-reform QBID. "
            "Without farm_rent_income the reform is a structural zero."
        ),
        "issue": "PolicyEngine/populace#298",
    },
    {
        "id": "domestic_production_ald_reactivation",
        "name": "Former Section 199 domestic-production deduction reactivation",
        "parameter_changes": {
            "gov.irs.ald.deductions": {
                "2024-01-01.2024-12-31": [
                    "loss_ald",
                    "self_employment_tax_ald",
                    "student_loan_interest_ald",
                    "early_withdrawal_penalty",
                    "alimony_expense_ald",
                    "educator_expense",
                    "health_savings_account_ald",
                    "self_employed_health_insurance_ald",
                    "self_employed_pension_contribution_ald",
                    "traditional_ira_contributions",
                    "qualified_adoption_assistance_expense",
                    "us_bonds_for_higher_ed",
                    "specified_possession_income",
                    "puerto_rico_income",
                    "domestic_production_ald",
                ]
            }
        },
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["domestic_production_ald"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "PolicyEngine-US 1.764.6 excludes the former Section 199 deduction "
            "from current-law above-the-line deductions. This probe preserves "
            "the exact 2024 list and adds only domestic_production_ald, so "
            "baseline-minus-reform income tax must be positive. Without the "
            "restored E03240 input, reactivation is a structural zero."
        ),
        "issue": "PolicyEngine/populace#298",
    },
    {
        "id": "form_4952_election_neutralization",
        "name": "Form 4952 elected investment income neutralization",
        "parameter_changes": {},
        "neutralized_variable": "investment_income_elected_form_4952",
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["investment_income_elected_form_4952"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "PolicyEngine-US subtracts the tax-unit sum of "
            "investment_income_elected_form_4952 from net capital gain. "
            "Neutralizing only that leaf increases preferential net capital "
            "gain and lowers income tax, so baseline-minus-reform income tax "
            "must be positive. Without the restored E58990 input the "
            "neutralization is a structural zero."
        ),
        "issue": "PolicyEngine/populace#274",
    },
    {
        "id": "child_support_received_snap_exclusion",
        "name": "Exclude child-support receipts from SNAP unearned income",
        "parameter_changes": {
            "gov.usda.snap.income.sources.unearned": {
                "2024-01-01.2024-12-31": [
                    "ssi",
                    "tanf",
                    "general_assistance",
                    "pension_income",
                    "veterans_benefits",
                    "unemployment_compensation",
                    "disability_benefits",
                    "workers_compensation",
                    "social_security",
                    "retirement_distributions",
                    "rental_income",
                    "alimony_income",
                    "financial_assistance",
                    "survivor_benefits",
                    "dividend_income",
                    "interest_income",
                    "miscellaneous_income",
                ]
            }
        },
        "budget_measure": "snap",
        "period": 2024,
        "effect_direction": "reform_minus_baseline",
        "expected_sign": "positive",
        "binding_inputs": ["child_support_received"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only child_support_received from SNAP unearned-income "
            "sources lowers countable income and must increase SNAP for some "
            "recipients. Without the measured/QRF child-support receipt leaf, "
            "the source-list reform is a structural zero."
        ),
        "issue": "PolicyEngine/populace#32",
    },
    {
        "id": "child_support_expense_snap_deduction_abolition",
        "name": "Abolish the SNAP child-support expense deduction",
        "parameter_changes": {
            "gov.usda.snap.income.deductions.allowed": {
                "2024-01-01.2024-12-31": [
                    "snap_standard_deduction",
                    "snap_earned_income_deduction",
                    "snap_dependent_care_deduction",
                    "snap_excess_medical_expense_deduction",
                    "snap_excess_shelter_expense_deduction",
                ]
            }
        },
        "budget_measure": "snap",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["child_support_expense"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only snap_child_support_deduction raises countable net "
            "income and must reduce SNAP in states that take the expense as a "
            "net-income deduction. Without the measured/QRF positive expense "
            "leaf, abolition is a structural zero."
        ),
        "issue": "PolicyEngine/populace#32",
    },
    {
        "id": "disability_benefits_snap_exclusion",
        "name": "Exclude disability benefits from SNAP unearned income",
        "parameter_changes": {
            "gov.usda.snap.income.sources.unearned": {
                "2024-01-01.2024-12-31": [
                    "ssi",
                    "tanf",
                    "general_assistance",
                    "pension_income",
                    "veterans_benefits",
                    "unemployment_compensation",
                    "workers_compensation",
                    "social_security",
                    "retirement_distributions",
                    "rental_income",
                    "child_support_received",
                    "alimony_income",
                    "financial_assistance",
                    "survivor_benefits",
                    "dividend_income",
                    "interest_income",
                    "miscellaneous_income",
                ]
            }
        },
        "budget_measure": "snap",
        "period": 2024,
        "effect_direction": "reform_minus_baseline",
        "expected_sign": "positive",
        "binding_inputs": ["disability_benefits"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only disability_benefits from SNAP unearned-income "
            "sources lowers countable income and must increase SNAP for some "
            "recipients. Without the measured/QRF non-workers-compensation "
            "benefit leaf, the source-list reform is a structural zero."
        ),
        "issue": "PolicyEngine/populace#38",
    },
    {
        "id": "educator_expense_ald_abolition",
        "name": "Abolish the educator-expense above-the-line deduction",
        "parameter_changes": {
            "gov.irs.ald.deductions": {
                "2024-01-01.2024-12-31": [
                    "loss_ald",
                    "self_employment_tax_ald",
                    "student_loan_interest_ald",
                    "early_withdrawal_penalty",
                    "alimony_expense_ald",
                    "health_savings_account_ald",
                    "self_employed_health_insurance_ald",
                    "self_employed_pension_contribution_ald",
                    "traditional_ira_contributions",
                    "qualified_adoption_assistance_expense",
                    "us_bonds_for_higher_ed",
                    "specified_possession_income",
                    "puerto_rico_income",
                ]
            }
        },
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "reform_minus_baseline",
        "expected_sign": "positive",
        "binding_inputs": ["educator_expense"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Removing only educator_expense from the above-the-line deduction "
            "source list raises taxable income and must increase income tax for "
            "some filers. Without the restored PUF E03220 leaf, abolition is a "
            "structural zero."
        ),
        "issue": "PolicyEngine/populace#32",
    },
    {
        "id": "alimony_expense_ald_abolition",
        "name": "Alimony expense above-the-line deduction abolition",
        "parameter_changes": {
            "gov.irs.ald.alimony_expense.divorce_year_threshold[0].amount": {
                "2024-01-01.2100-12-31": False
            }
        },
        "budget_measure": "income_tax",
        "period": 2024,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["alimony_expense"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "The retired export has no nondefault divorce_year input, so "
            "PolicyEngine-US applies its default year 0 through the first "
            "eligibility bracket. Setting that bracket's amount to false "
            "abolishes the alimony-expense above-the-line deduction on the "
            "release, so baseline-minus-reform income tax must be negative. "
            "With alimony_expense absent or degenerate, the abolition scores "
            "exactly $0."
        ),
        "issue": "PolicyEngine/populace#38",
    },
    {
        "id": "obbba_casualty_loss_limit",
        "name": "OBBBA casualty-loss deduction reactivation",
        "parameter_changes": {
            "gov.irs.deductions.itemized.casualty.active": {
                "2026-01-01.2026-12-31": True
            }
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["casualty_loss"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Reactivating the casualty-loss deduction lowers income tax only "
            "for tax units with casualty_loss above the statutory AGI floor, "
            "so baseline-minus-reform income tax must be positive. With the "
            "casualty-loss input absent or degenerate, the reactivation scores "
            "exactly $0."
        ),
        "issue": "PolicyEngine/populace#32",
    },
    {
        "id": "obbba_misc_itemized_deductions",
        "name": "OBBBA miscellaneous-itemized deduction reactivation",
        "parameter_changes": {
            "gov.irs.deductions.itemized.misc.applies": {"2026-01-01.2026-12-31": True}
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": ["unreimbursed_business_employee_expenses"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Reactivating the miscellaneous itemized deduction lowers income "
            "tax only for tax units with qualifying expenses above its AGI "
            "floor, so baseline-minus-reform income tax must be positive. "
            "Without unreimbursed_business_employee_expenses, the retired "
            "pipeline's only populated miscellaneous-expense input, the "
            "reactivation is a structural zero on the export."
        ),
        "issue": "PolicyEngine/populace#32",
    },
    {
        "id": "obbba_cdcc",
        "name": "OBBBA Child and Dependent Care Credit reversion",
        "parameter_changes": {
            "gov.irs.credits.cdcc.phase_out.max": {"2026-01-01.2026-12-31": 0.35},
            "gov.irs.credits.cdcc.phase_out.min": {"2026-01-01.2026-12-31": 0.2},
            "gov.irs.credits.cdcc.phase_out.amended_structure.applies": {
                "2026-01-01.2026-12-31": False
            },
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["spm_unit_pre_subsidy_childcare_expenses"],
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Reverting the OBBBA CDCC enhancement raises income tax, so "
            "baseline-minus-reform income tax must be negative. With measured "
            "pre-subsidy childcare expenses absent or degenerate, no filer has "
            "qualifying care expenses and the reversion scores exactly $0."
        ),
        "issue": "PolicyEngine/populace#278",
    },
    {
        "id": "obbba_no_tax_on_tips",
        "name": "OBBBA no-tax-on-tips deduction",
        "parameter_changes": {
            "gov.irs.deductions.tip_income.cap": {"2026-01-01.2026-12-31": 0}
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": [
            "tip_income",
            "treasury_tipped_occupation_code",
        ],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting the OBBBA tip-deduction cap to zero removes the deduction, "
            "so baseline-minus-reform income tax must be negative in 2026. "
            "With tip_income or the Treasury tipped-occupation code absent, "
            "qualified tip income is zero and the repeal scores exactly $0."
        ),
        "issue": "PolicyEngine/populace#38",
    },
    {
        "id": "obbba_no_tax_on_overtime",
        "name": "OBBBA no-tax-on-overtime deduction",
        "parameter_changes": {
            "gov.irs.deductions.overtime_income.cap.JOINT": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.SINGLE": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.HEAD_OF_HOUSEHOLD": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.SURVIVING_SPOUSE": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.SEPARATE": {
                "2026-01-01.2026-12-31": 0
            },
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["fsla_overtime_premium"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting every OBBBA overtime-deduction cap to zero removes the "
            "deduction, so reform income tax rises and baseline-minus-reform "
            "must be negative in 2026. With fsla_overtime_premium absent or "
            "degenerate, qualified overtime is zero and the repeal scores $0."
        ),
        "issue": "PolicyEngine/populace#242",
    },
    {
        "id": "obbba_auto_loan_interest",
        "name": "OBBBA no-tax-on-auto-loan-interest deduction",
        "parameter_changes": {
            "gov.irs.deductions.auto_loan_interest.cap": {"2026-01-01.2026-12-31": 0}
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["qualified_passenger_vehicle_loan_interest"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting the OBBBA auto-loan-interest deduction cap to zero "
            "removes the deduction, so reform income tax rises and "
            "baseline-minus-reform must be negative in 2026. With qualified "
            "passenger-vehicle loan interest absent or degenerate, the repeal "
            "scores exactly $0."
        ),
        "issue": "PolicyEngine/populace#252",
    },
    {
        "id": "tx_snap_additional_vehicle_exemption_abolition",
        "name": "Texas SNAP additional-vehicle exemption abolition",
        "parameter_changes": {
            "gov.hhs.tanf.non_cash.tx_additional_vehicle_exemption": {
                "2026-01-01.2100-12-31": 0
            }
        },
        "budget_measure": "snap",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "positive",
        "binding_inputs": list(SIPP_VEHICLE_INPUTS),
        "min_abs_effect": 1_000_000.0,
        "reason": (
            "Setting the Texas TANF non-cash additional-vehicle exemption "
            "to zero tightens the asset test used by Texas SNAP categorical "
            "eligibility, so baseline-minus-reform SNAP must be positive. "
            "PolicyEngine-US computes the exemption from both household "
            "vehicle count and value; if either restored SIPP vehicle input "
            "is absent or degenerate, this vehicle-specific reform loses its "
            "intended binding channel. A persisted 30,000-household Populace "
            "smoke scored +$3.58 million in 2026 SNAP."
        ),
        "issue": "PolicyEngine/populace#49",
    },
]


def _load(name: str) -> dict:
    return json.loads((US_PACKAGE_DIR / name).read_text(encoding="utf-8"))


def build_manifest() -> dict:
    parity = _load("ecps_parity_reference.json")
    known_gaps = _load("ecps_parity_known_gaps.json")["known_gaps"]

    populated_layers = {
        name for name, share in parity["nonzero_shares"].items() if float(share) > 0.0
    } | set(POST_REFERENCE_ECPS_REQUIRED_INPUTS)
    ssi_assets = set(SSI_COUNTABLE_RESOURCE_ASSETS)

    missing_assets = sorted(ssi_assets - populated_layers)
    if missing_assets:
        raise ValueError(
            "SSI countable-resource asset inputs are not in the reference eCPS "
            f"populated surface, cannot pin them as required: {missing_assets}."
        )
    restored_inputs = set(RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS)
    missing_restored = sorted(restored_inputs - populated_layers)
    if missing_restored:
        raise ValueError(
            "Restored reference inputs are absent from the reference populated "
            f"surface: {missing_restored}."
        )
    stale_restored_gaps = sorted(restored_inputs & set(known_gaps))
    if stale_restored_gaps:
        raise ValueError(
            "Restored reference inputs cannot remain in the parity-gap register: "
            f"{stale_restored_gaps}."
        )

    columns: dict[str, dict] = {}
    for name in sorted(populated_layers):
        if name in known_gaps and name not in ssi_assets:
            entry = known_gaps[name]
            columns[name] = {
                "status": "reviewed_exclusion",
                "reason": str(entry["reason"]),
                "issue": str(entry["issue"]),
            }
        else:
            column = {"status": "required"}
            if name in ssi_assets:
                # These are currently absent; #368 forbids an exclusion so the
                # gate fails until Build J restores them.
                column["note"] = (
                    "SSI countable-resource asset input; required with NO "
                    "reviewed exclusion per PolicyEngine/populace#368 so the "
                    "gate fails until the asset stage is restored (Deliverable "
                    "2). Currently absent — this is the intended red gate."
                )
            columns[name] = column

    required = sorted(n for n, c in columns.items() if c["status"] == "required")
    reviewed = sorted(
        n for n, c in columns.items() if c["status"] == "reviewed_exclusion"
    )

    # Provenance without naming the retired data package. Only the eCPS parity
    # reference, its loader, and its test are allow-listed to name the incumbent
    # (test_us_plan.test_no_incumbent_data_package_references_in_live_tree); this
    # manifest is not, so it records the sha-locked coordinates that DON'T name a
    # package (filename + content sha + revision + vintage) and points at that
    # allow-listed parity reference for the full record. Nothing reads these
    # fields — the gate and the anti-rot check derive the surface from
    # ecps_parity_reference.json directly — so this stays pure documentation.
    parity_source = dict(parity["source"])
    reference = {
        "derived_from": "ecps_parity_reference.json",
        "filename": str(parity_source.get("filename", "")),
        "revision": str(parity_source.get("revision", "")),
        "sha256": str(parity_source.get("sha256", "")),
        "vintage": str(parity_source.get("vintage", "")),
        "period": str(parity_source.get("period", "")),
        "note": (
            "Column surface derived from the populated input layers of the "
            "pinned, sha-verified reference eCPS recorded in "
            "ecps_parity_reference.json (the launch parity contract, "
            "PolicyEngine/populace#313) — the single allow-listed record of the "
            "incumbent coordinates. This manifest names no data package."
        ),
    }

    return {
        "schema_version": 1,
        "issue": "PolicyEngine/populace#368",
        "description": (
            "Declared full-coverage contract for a US release: every input "
            "column the reference eCPS exports must be persisted as a key with "
            "non-default signal, or carry a reviewed exclusion. Enforced as a "
            "hard release gate (populace.build.us_runtime.release_input_"
            "coverage) that generalizes assert_required_us_release_source_"
            "columns from 5 columns to the full eCPS input surface."
        ),
        "reference": reference,
        "derivation": (
            "Required surface = input columns in the pinned, sha-verified "
            "ecps_parity_reference.json populated layers, plus the documented "
            "post-reference fsla_overtime_premium, "
            "qualified_passenger_vehicle_loan_interest, and five desired "
            "retirement-contribution inputs required by shipped validation "
            "probes. "
            "status='reviewed_exclusion' for ecps_parity_known_gaps.json entries "
            "(reason+issue from that register); EXCEPT every primary-source "
            "restoration pinned by RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS "
            "(including the Section 199A QBI family), and the SSI countable-"
            "resource asset inputs (bank_account_assets, "
            "stock_assets, bond_assets), which are status='required' with NO "
            "exclusion per PolicyEngine/populace#368 "
            "so the gate fails on today's artifacts and asset restoration "
            "(Deliverable 2) turns it green. All other populated layers are "
            "'required'. Regenerate with "
            "tools/build_us_release_input_coverage_manifest.py."
        ),
        "counts": {
            "required": len(required),
            "reviewed_exclusion": len(reviewed),
            "total": len(columns),
        },
        "ssi_countable_resource_assets": list(SSI_COUNTABLE_RESOURCE_ASSETS),
        "columns": columns,
        "reform_coverage_probes": REFORM_COVERAGE_PROBES,
    }


def main() -> None:
    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {MANIFEST_PATH} — {manifest['counts']['required']} required, "
        f"{manifest['counts']['reviewed_exclusion']} reviewed exclusions, "
        f"{len(manifest['reform_coverage_probes'])} reform probe(s)."
    )


if __name__ == "__main__":
    main()
