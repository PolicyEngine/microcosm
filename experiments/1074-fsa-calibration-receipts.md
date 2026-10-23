# microcosm#1074 receipts: Federal Student Aid portfolio targets

Generated 2026-10-22 by `experiments/fit_1074_fsa_calibration.py`, commit `c94516ee14b1920a490902288115d66dbc743762`, Modal run `20261022T194252Z-c94516ee1` (`experiments/modal_1074.py`). Machine-readable copy: `experiments/receipts/1074-fsa-calibration-fit.json`.

Host: the published default `populace-us-2024-spm-20260915` (sha256 `6496cc4393d4…`, 57,240 households), with the SCF 2025 education-debt stage run at seed 0. Feed: the pinned `51aa40f` US feed plus the draft chronicle#347 FSA rows (`d8e3d0fd9e21…`, 40,639 rows). Arms: **release** = the published weights; **A** = a dense refit to the `national_state` surface without the FSA family; **B** = the same refit with it. Same epochs (1500), learning rate, 5x weight cap and conserved mass as a release. Not a release: no gate ran and nothing was published.

## FSA targets: the reconciled levels

| | Balance | Borrowers |
|---|---:|---:|
| 50 states and DC, as published | $1,627.4B | 42.38M |
| Puerto Rico and Other (leave) | $16.7B | 433,400 |
| Not Reported (allocated) | $52.2B | 1.95M |
| Location table total | $1,696.3B | 44.76M |
| Allocation factor | 1.0317 | 1.0456 |
| In-scope control (every family sums to it) | $1,679.1B | 44.31M |

Reference date: Data as of Mar. 31, 2026 (`2026-03`), the latest quarter all four bound tables publish; unaged at the 2024 build period (`period_alignment = stock_at_source_reference_date`).

## Fit by FSA family

Mean absolute relative error (share of targets within 5%).

| FSA family | Targets | Release | A, no FSA | B, with FSA | A + L2 | B + L2 |
|---|---:|---:|---:|---:|---:|---:|
| fsa_age_band_balance | 5 | 50.3% (0%) | 48.4% (0%) | 0.1% (100%) | 50.6% (0%) | 1.1% (100%) |
| fsa_age_band_borrowers | 5 | 53.1% (0%) | 52.1% (0%) | 10.7% (40%) | 52.8% (0%) | 9.5% (80%) |
| fsa_balance_excluding_parent_plus | 1 | 54.3% (0%) | 53.1% (0%) | 0.1% (100%) | 54.5% (0%) | 1.5% (100%) |
| fsa_debt_size_band_balance | 9 | 51.0% (0%) | 50.3% (0%) | 1.0% (89%) | 51.1% (0%) | 4.8% (78%) |
| fsa_debt_size_band_borrowers | 9 | 49.6% (0%) | 49.4% (0%) | 3.7% (78%) | 49.9% (0%) | 6.8% (67%) |
| fsa_parent_plus_balance | 1 | 25.3% (0%) | 32.8% (0%) | 0.0% (100%) | 25.2% (0%) | 10.3% (0%) |
| fsa_state_balance | 51 | 47.0% (0%) | 46.7% (2%) | 0.2% (100%) | 47.0% (0%) | 1.1% (92%) |
| fsa_state_borrowers | 51 | 49.2% (0%) | 49.8% (0%) | 1.7% (90%) | 49.1% (0%) | 7.8% (59%) |

## Headline levels

| | FSA target | Release | A, no FSA | B, with FSA | A + L2 | B + L2 |
|---|---:|---:|---:|---:|---:|---:|
| Federal balance, own plus Parent PLUS | $1,679.1B | $858.4B | $886.0B | $1,678.0B | $855.8B | $1,668.1B |
| Borrowers (people with a balance) | 44.31M | 21.22M | 21.45M | 43.78M | 21.43M | 41.95M |
| Parent PLUS balance | $114.8B | $143.9B | $152.6B | $114.9B | $143.8B | $126.7B |
| Parent PLUS share of the balance | 6.8% | 16.8% | 17.2% | 6.8% | 16.8% | 7.6% |
| Mean balance per borrower | $37,894 | $40,447 | $41,313 | $38,331 | $39,934 | $39,767 |
| Households with federal education debt |  | 16.4% | 16.6% | 33.0% | 16.6% | 31.5% |

## Every other family: does adding FSA cost the rest of the surface?

Mean absolute relative error over the `national_state` surface's non-FSA targets, by provider/category (the dashboard's family), each miss capped at 100% of its target as the loss caps it, sorted by how much adding the FSA family moves it:

| Family | Targets | Release | A, no FSA | B, with FSA | A + L2 | B + L2 | B - A (pp) | B+L2 - A+L2 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hhs_acf_tanf/hhs_acf_tanf.hhs_acf_tanf | 30 | 29.8% | 18.3% | 20.5% | 22.5% | 22.4% | +2.17 | -0.05 |
| jct/jct.individual_tax_expenditure_revenue_loss | 11 | 42.3% | 35.3% | 37.5% | 39.9% | 39.3% | +2.14 | -0.63 |
| bea/bea.nipa_proprietors_income | 1 | 11.6% | 5.9% | 7.8% | 11.2% | 9.5% | +1.90 | -1.72 |
| ssa/ssa.ssi_federal_payment_recipients | 3 | 28.5% | 15.6% | 17.2% | 24.3% | 21.6% | +1.60 | -2.68 |
| irs_soi/irs_soi.capital_gains_gross | 104 | 30.1% | 17.0% | 18.5% | 26.0% | 24.6% | +1.52 | -1.43 |
| irs_soi/irs_soi.assigned_aca_ptc | 104 | 0.9% | 0.6% | 1.6% | 0.7% | 1.2% | +1.04 | +0.45 |
| irs_soi/irs_soi.real_estate_taxes | 106 | 6.5% | 5.3% | 6.3% | 6.3% | 6.4% | +0.95 | +0.14 |
| irs_soi/irs_soi.refundable_ctc | 104 | 0.8% | 0.3% | 1.2% | 0.7% | 1.3% | +0.87 | +0.61 |
| irs_soi/irs_soi.eitc | 781 | 3.1% | 2.5% | 3.3% | 2.7% | 3.1% | +0.84 | +0.43 |
| irs_soi/irs_soi.ctc | 104 | 1.5% | 0.6% | 1.3% | 1.2% | 1.4% | +0.67 | +0.19 |
| irs_soi/irs_soi.charitable_deduction | 2 | 7.5% | 6.8% | 7.4% | 6.2% | 6.7% | +0.58 | +0.56 |
| irs_soi/irs_soi.salt_deduction | 106 | 1.9% | 1.8% | 2.3% | 2.1% | 2.3% | +0.51 | +0.20 |
| irs_soi/irs_soi.interest_deduction | 2 | 2.0% | 1.7% | 1.2% | 2.4% | 2.0% | -0.49 | -0.43 |
| irs_soi/irs_soi.taxable_interest_income | 1038 | 6.0% | 5.7% | 6.2% | 6.0% | 5.9% | +0.49 | -0.08 |
| irs_soi/irs_soi.unemployment_compensation | 106 | 0.1% | 0.3% | 0.7% | 0.2% | 0.7% | +0.45 | +0.45 |
| irs_soi/irs_soi.count | 260 | 31.7% | 20.5% | 20.9% | 22.9% | 22.1% | +0.37 | -0.79 |
| irs_soi/irs_soi.qualified_dividends | 104 | 0.3% | 0.5% | 0.9% | 0.5% | 0.5% | +0.34 | +0.05 |
| ssa/ssa.ssi_payments | 51 | 3.7% | 0.4% | 0.6% | 0.5% | 0.9% | +0.18 | +0.31 |
| irs_soi/irs_soi.adjusted_gross_income | 261 | 19.7% | 0.6% | 0.7% | 2.8% | 2.0% | +0.16 | -0.76 |
| irs_soi/irs_soi.income_tax | 104 | 0.6% | 0.7% | 0.8% | 0.8% | 0.9% | +0.15 | +0.08 |
| irs_soi/irs_soi.income_tax_before_credits | 104 | 1.6% | 1.6% | 1.8% | 1.9% | 2.0% | +0.15 | +0.08 |
| cms_medicare/cms_medicare.cms_medicare | 1 | 13.8% | 13.5% | 13.6% | 13.5% | 13.3% | +0.15 | -0.23 |
| irs_soi/irs_soi.estate_losses | 2 | 50.0% | 50.2% | 50.1% | 50.2% | 50.3% | -0.12 | +0.11 |
| irs_soi/irs_soi.ordinary_dividend_income | 108 | 0.8% | 0.8% | 0.9% | 0.8% | 0.8% | +0.12 | -0.06 |
| usda_snap/usda_snap.usda_snap | 104 | 7.9% | 0.1% | 0.2% | 0.6% | 0.5% | +0.12 | -0.15 |
| ssa/ssa.ssi_recipients | 51 | 4.8% | 0.7% | 0.8% | 2.0% | 1.5% | +0.11 | -0.48 |
| census_stc/census_stc.individual_income_tax_collections | 44 | 13.9% | 13.7% | 13.8% | 13.9% | 13.9% | +0.11 | -0.01 |
| irs_soi/irs_soi.taxable_income | 107 | 0.3% | 0.4% | 0.5% | 0.6% | 0.6% | +0.11 | -0.00 |
| bea/bea.nipa_wages_and_salaries | 1 | 13.3% | 13.6% | 13.5% | 13.5% | 13.6% | -0.10 | +0.18 |
| irs_soi/irs_soi.medical_expense_deduction | 104 | 6.0% | 6.3% | 6.3% | 6.2% | 6.4% | +0.09 | +0.24 |
| irs_soi/irs_soi.tax_exempt_interest_income | 104 | 0.1% | 0.7% | 0.8% | 1.0% | 1.0% | +0.09 | -0.06 |
| irs_soi/irs_soi.employment_income | 106 | 0.1% | 0.1% | 0.2% | 0.2% | 0.2% | +0.08 | -0.00 |
| ssa/ssa.ssa | 6 | 3.7% | 0.1% | 0.0% | 1.4% | 1.1% | -0.08 | -0.38 |
| irs_soi/irs_soi.estate_income | 2 | 0.0% | 0.1% | 0.0% | 0.1% | 0.1% | -0.07 | +0.06 |
| irs_soi/irs_soi.non_sch_d_capital_gains | 2 | 0.0% | 0.0% | 0.1% | 0.1% | 0.1% | +0.07 | -0.01 |
| irs_soi/irs_soi.deductible_mortgage_interest | 2 | 0.4% | 0.0% | 0.1% | 0.1% | 0.0% | +0.07 | -0.08 |
| cbo/cbo.cbo | 5 | 4.5% | 4.6% | 4.7% | 4.6% | 4.7% | +0.05 | +0.18 |
| irs_soi/irs_soi.tip_income | 1 | 21.3% | 0.1% | 0.2% | 0.1% | 0.0% | +0.04 | -0.04 |
| irs_soi/irs_soi.partnership_and_s_corp_income | 104 | 0.6% | 0.4% | 0.3% | 0.6% | 0.6% | -0.04 | -0.05 |
| irs_soi/irs_soi.itemized_taxable_income_deductions | 55 | 2.0% | 2.3% | 2.3% | 2.5% | 2.3% | +0.03 | -0.23 |
| cms_medicaid/cms_medicaid.cms_medicaid | 135 | 3.1% | 2.3% | 2.3% | 2.3% | 2.3% | +0.03 | +0.01 |
| irs_soi/irs_soi.rent_and_royalty_net_income | 104 | 4.0% | 3.1% | 3.1% | 3.1% | 3.1% | +0.02 | -0.06 |
| hhs_acf_liheap/hhs_acf_liheap.hhs_acf_liheap | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | +0.02 | +0.08 |
| cms_aca/cms_aca.cms_aca | 102 | 1.6% | 0.2% | 0.2% | 0.7% | 0.6% | +0.02 | -0.13 |
| federal_reserve/federal_reserve.federal_reserve | 1 | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% | -0.01 | -0.01 |
| irs_soi/irs_soi.business_net_profits | 104 | 1.3% | 1.1% | 1.1% | 1.1% | 1.1% | +0.01 | -0.00 |
| irs_soi/irs_soi.ira_distributions | 108 | 0.1% | 0.1% | 0.1% | 0.1% | 0.2% | -0.00 | +0.01 |
| census_pep/census_pep.resident_population | 936 | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | +0.00 | +0.01 |
| irs_soi/irs_soi.taxable_social_security | 108 | 0.3% | 0.2% | 0.2% | 0.3% | 0.3% | +0.00 | +0.05 |
| irs_soi/irs_soi.taxable_pension_income | 108 | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | +0.00 | +0.01 |
| **all non-FSA targets** | 6102 | 5.3% | 3.3% | 3.7% | 3.8% | 3.8% | +0.36 | -0.02 |

## Racial composition of borrowers, before and after calibration

Borrower households by the reference person's race (the SCF's `racecl4`; the SCF column pools the five implicates):

| | SCF 2025 | Release | A, no FSA | B, with FSA | A + L2 | B + L2 | B - release (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| white_non_hispanic | 57.6% | 60.8% | 60.0% | 57.9% | 62.1% | 60.0% | -2.9 |
| black_non_hispanic | 18.6% | 17.4% | 16.4% | 19.0% | 16.8% | 19.0% | +1.6 |
| hispanic | 9.2% | 12.9% | 13.8% | 14.8% | 12.4% | 12.8% | +1.9 |
| other_or_multiple | 14.7% | 9.0% | 9.8% | 8.3% | 8.7% | 8.2% | -0.7 |

Borrowers (people) by their own race, and the share of adults who borrow:

| | Release | A, no FSA | B, with FSA | A + L2 | B + L2 | Adults with a balance: release | B | B + L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| white_non_hispanic | 61.5% | 60.9% | 59.2% | 62.9% | 61.4% | 7.9% | 16.0% | 15.0% |
| black_non_hispanic | 16.4% | 15.7% | 17.8% | 16.0% | 17.7% | 12.8% | 26.2% | 25.3% |
| hispanic | 13.0% | 13.5% | 14.5% | 12.2% | 12.6% | 6.2% | 14.2% | 14.1% |
| other_or_multiple | 9.0% | 9.9% | 8.4% | 8.8% | 8.3% | 7.6% | 13.9% | 13.8% |

## Weight movement against the published weights

| | A, no FSA | B, with FSA | A + L2 | B + L2 |
|---|---:|---:|---:|---:|
| Borrower households' weight ratio | 1.01 | 2.01 | 1.01 | 1.92 |
| Other households' weight ratio | 0.998 | 0.801 | 0.998 | 0.819 |
| Borrower households' share of weight | 16.6% | 33.0% | 16.6% | 31.5% |
| Median household weight ratio | 0.02 | 4.4e-04 | 0.13 | 3.9e-03 |
| 99th percentile weight ratio | 4.76 | 4.98 | 1.97 | 2.80 |
| Effective sample size (release 9,174) | 4,149 | 2,951 | 6,435 | 4,248 |

## FSA facts that do not bind

| Disposition | Facts | Reason |
|---|---:|---|
| allocated | 2 | 'Not Reported' (FFEL and Perkins loans whose commercial lenders need not report an address) has no state; it is allocated to every reported location in proportion to that location's reported value. |
| allocated | 2 | Age not reported: in no partition, so the reported bands, which bind as shares of the in-scope control, absorb it pro rata. |
| diagnostic | 12 | ED-held FFEL IBR by partial financial hardship: no plan or hardship column in the build; diagnostic only. |
| diagnostic | 45 | ED-held FFEL by repayment plan: no plan column in the build; diagnostic only. |
| diagnostic | 24 | Federally managed IBR/PAYE by partial financial hardship: no plan or hardship column in the build; diagnostic only. |
| diagnostic | 65 | Federally managed portfolio by repayment plan: no plan column in the build; diagnostic only. |
| diagnostic | 24 | Income-driven plans by partial financial hardship: no plan or hardship column in the build; diagnostic only. |
| diagnostic | 24 | Recipient-based totals by loan program (Direct Loan, FFEL, Perkins). The stage's balances do not record the program, and FSA counts the student, not the parent, as the recipient of a Parent PLUS loan; the borrower-based location table sets the level instead. |
| diagnostic | 57 | Repayment plan: diagnostic only. The SCF records only an income-driven yes/no, the build carries no plan column, and FSA's recipient counts are loan-level (microcosm#1074). |
| excluded | 6 | Loan-type recipient counts are loan-level (a borrower with two loan types counts twice), so they are not counts of people. |
| excluded | 2 | Stafford (Combined) is a subtotal of rows the loan-type targets already count. |
| other_quarter | 58 | Not the reference quarter: every bound table binds at the latest quarter all four publish (2026-03). |
| out_of_scope | 2 | 'Other' is a borrower address outside the states and territories the table lists, outside the host universe; it leaves the in-scope control with its pro-rata share of Not Reported. |
| out_of_scope | 2 | Puerto Rico is outside the host universe (the CPS samples the 50 states and DC). Its rows, with their pro-rata share of Not Reported, leave the in-scope control. |

Materialization dropped 0 FSA targets.

## Every FSA target

| Target | FSA (reconciled) | Release | A, no FSA | B, with FSA | A + L2 | B + L2 |
|---|---:|---:|---:|---:|---:|---:|
| Federal student loan balance, Alabama | $26.8B | $8.3B (-69.0%) | $7.3B (-72.7%) | $26.8B (+0.0%) | $7.9B (-70.7%) | $26.8B (-0.2%) |
| Federal student loan balance, Alaska | $2.5B | $1.1B (-53.6%) | $1.4B (-42.4%) | $2.5B (-0.0%) | $1.2B (-52.7%) | $2.3B (-9.1%) |
| Federal student loan balance, Arizona | $35.1B | $24.3B (-30.7%) | $14.4B (-59.0%) | $35.1B (-0.1%) | $18.5B (-47.1%) | $35.1B (-0.0%) |
| Federal student loan balance, Arkansas | $15.0B | $6.0B (-59.9%) | $6.1B (-59.3%) | $15.0B (+0.1%) | $6.3B (-57.6%) | $15.0B (+0.0%) |
| Federal student loan balance, California | $161.7B | $91.3B (-43.5%) | $99.2B (-38.6%) | $161.8B (+0.1%) | $96.8B (-40.1%) | $161.6B (-0.0%) |
| Federal student loan balance, Colorado | $30.6B | $12.2B (-60.0%) | $6.4B (-79.1%) | $30.6B (-0.0%) | $9.0B (-70.5%) | $30.6B (-0.2%) |
| Federal student loan balance, Connecticut | $20.4B | $8.2B (-59.8%) | $8.1B (-60.2%) | $20.4B (-0.2%) | $7.9B (-61.1%) | $20.4B (-0.1%) |
| Federal student loan balance, Delaware | $5.8B | $2.6B (-54.7%) | $3.3B (-42.6%) | $5.8B (-0.2%) | $2.9B (-49.4%) | $5.4B (-5.8%) |
| Federal student loan balance, District of Columbia | $6.6B | $2.6B (-60.6%) | $1.5B (-77.5%) | $6.6B (+0.1%) | $2.5B (-62.1%) | $6.6B (-0.3%) |
| Federal student loan balance, Florida | $116.7B | $42.9B (-63.2%) | $43.9B (-62.4%) | $116.7B (-0.0%) | $42.8B (-63.4%) | $116.8B (+0.1%) |
| Federal student loan balance, Georgia | $77.0B | $38.4B (-50.1%) | $51.9B (-32.5%) | $77.1B (+0.1%) | $33.3B (-56.8%) | $76.8B (-0.2%) |
| Federal student loan balance, Hawaii | $5.1B | $3.3B (-35.6%) | $3.2B (-36.0%) | $5.1B (-0.1%) | $3.2B (-36.7%) | $5.1B (-0.1%) |
| Federal student loan balance, Idaho | $7.7B | $6.2B (-20.5%) | $6.4B (-17.2%) | $7.7B (+0.1%) | $6.5B (-16.0%) | $7.7B (-0.2%) |
| Federal student loan balance, Illinois | $67.4B | $34.1B (-49.3%) | $50.4B (-25.2%) | $67.4B (+0.1%) | $47.7B (-29.2%) | $67.3B (-0.1%) |
| Federal student loan balance, Indiana | $32.0B | $25.3B (-20.8%) | $26.4B (-17.3%) | $32.0B (+0.0%) | $28.2B (-11.8%) | $32.0B (-0.0%) |
| Federal student loan balance, Iowa | $13.9B | $5.5B (-60.6%) | $6.5B (-53.1%) | $14.0B (+0.2%) | $6.1B (-56.5%) | $13.9B (+0.2%) |
| Federal student loan balance, Kansas | $13.5B | $6.5B (-52.1%) | $6.5B (-52.1%) | $13.5B (-0.0%) | $6.9B (-49.2%) | $13.3B (-1.7%) |
| Federal student loan balance, Kentucky | $22.3B | $12.8B (-42.6%) | $20.2B (-9.5%) | $22.3B (+0.0%) | $13.2B (-40.6%) | $22.3B (+0.2%) |
| Federal student loan balance, Louisiana | $25.9B | $6.1B (-76.6%) | $9.0B (-65.4%) | $24.6B (-5.0%) | $6.6B (-74.4%) | $19.2B (-25.9%) |
| Federal student loan balance, Maine | $6.9B | $4.0B (-42.0%) | $2.7B (-60.9%) | $6.9B (-0.0%) | $3.6B (-48.5%) | $6.9B (-0.1%) |
| Federal student loan balance, Maryland | $39.6B | $14.4B (-63.8%) | $15.2B (-61.5%) | $39.6B (-0.0%) | $14.4B (-63.6%) | $37.5B (-5.4%) |
| Federal student loan balance, Massachusetts | $34.5B | $19.3B (-44.0%) | $16.5B (-52.3%) | $34.5B (+0.2%) | $18.7B (-45.8%) | $34.4B (-0.1%) |
| Federal student loan balance, Michigan | $54.9B | $34.0B (-38.1%) | $39.5B (-28.0%) | $55.1B (+0.3%) | $34.5B (-37.1%) | $54.8B (-0.1%) |
| Federal student loan balance, Minnesota | $28.4B | $20.9B (-26.4%) | $21.5B (-24.1%) | $28.4B (+0.0%) | $22.3B (-21.3%) | $28.3B (-0.2%) |
| Federal student loan balance, Mississippi | $18.4B | $7.9B (-57.1%) | $6.3B (-65.8%) | $18.4B (+0.1%) | $7.9B (-56.9%) | $18.4B (-0.0%) |
| Federal student loan balance, Missouri | $31.6B | $15.7B (-50.2%) | $24.7B (-21.7%) | $31.6B (+0.0%) | $18.4B (-41.7%) | $31.6B (+0.2%) |
| Federal student loan balance, Montana | $4.6B | $3.6B (-23.5%) | $3.5B (-25.1%) | $4.6B (+0.1%) | $3.6B (-22.8%) | $4.7B (+0.2%) |
| Federal student loan balance, Nebraska | $8.5B | $3.9B (-54.3%) | $4.3B (-49.1%) | $8.5B (+0.1%) | $3.7B (-56.7%) | $8.5B (+0.2%) |
| Federal student loan balance, Nevada | $13.6B | $5.4B (-60.4%) | $6.8B (-50.2%) | $13.6B (-0.1%) | $6.0B (-55.9%) | $13.6B (-0.0%) |
| Federal student loan balance, New Hampshire | $7.0B | $3.6B (-48.4%) | $2.9B (-58.9%) | $7.0B (+0.2%) | $3.1B (-55.9%) | $7.0B (+0.3%) |
| Federal student loan balance, New Jersey | $49.9B | $22.9B (-54.1%) | $27.5B (-44.9%) | $49.9B (+0.0%) | $24.2B (-51.5%) | $50.0B (+0.0%) |
| Federal student loan balance, New Mexico | $8.3B | $9.6B (+15.8%) | $4.6B (-44.1%) | $8.2B (-0.2%) | $7.4B (-10.5%) | $8.3B (+0.2%) |
| Federal student loan balance, New York | $103.1B | $59.0B (-42.7%) | $52.1B (-49.4%) | $103.0B (-0.0%) | $54.5B (-47.1%) | $103.2B (+0.1%) |
| Federal student loan balance, North Carolina | $57.6B | $30.7B (-46.7%) | $48.5B (-15.8%) | $57.5B (-0.1%) | $34.4B (-40.2%) | $57.6B (+0.1%) |
| Federal student loan balance, North Dakota | $2.8B | $1.5B (-47.1%) | $1.6B (-42.6%) | $2.8B (-0.2%) | $1.6B (-44.3%) | $2.8B (-0.3%) |
| Federal student loan balance, Ohio | $66.3B | $24.9B (-62.4%) | $17.5B (-73.7%) | $66.4B (+0.0%) | $22.7B (-65.7%) | $66.2B (-0.1%) |
| Federal student loan balance, Oklahoma | $17.6B | $12.7B (-28.1%) | $10.4B (-41.1%) | $17.7B (+0.1%) | $11.9B (-32.6%) | $17.6B (+0.0%) |
| Federal student loan balance, Oregon | $21.5B | $9.8B (-54.5%) | $10.2B (-52.6%) | $21.5B (-0.0%) | $9.8B (-54.5%) | $21.0B (-2.0%) |
| Federal student loan balance, Pennsylvania | $71.8B | $27.9B (-61.1%) | $19.5B (-72.9%) | $71.7B (-0.1%) | $24.6B (-65.7%) | $71.7B (-0.1%) |
| Federal student loan balance, Rhode Island | $5.3B | $2.8B (-46.9%) | $2.2B (-58.6%) | $5.3B (+0.0%) | $2.7B (-48.8%) | $5.3B (+0.2%) |
| Federal student loan balance, South Carolina | $32.4B | $17.5B (-46.1%) | $14.4B (-55.5%) | $32.4B (-0.1%) | $13.3B (-58.9%) | $32.4B (+0.0%) |
| Federal student loan balance, South Dakota | $3.8B | $2.5B (-33.3%) | $1.2B (-67.7%) | $3.8B (+0.0%) | $1.8B (-52.5%) | $3.8B (+0.1%) |
| Federal student loan balance, Tennessee | $35.7B | $30.7B (-14.0%) | $34.8B (-2.6%) | $35.8B (+0.2%) | $30.9B (-13.5%) | $35.7B (-0.1%) |
| Federal student loan balance, Texas | $142.7B | $69.0B (-51.6%) | $44.8B (-68.6%) | $142.5B (-0.1%) | $58.3B (-59.2%) | $142.2B (-0.4%) |
| Federal student loan balance, Utah | $11.5B | $7.1B (-38.2%) | $9.2B (-19.8%) | $11.5B (+0.0%) | $8.6B (-24.6%) | $11.4B (-0.3%) |
| Federal student loan balance, Vermont | $3.0B | $1.5B (-49.1%) | $1.4B (-51.9%) | $3.0B (-0.1%) | $1.6B (-48.0%) | $3.0B (-0.1%) |
| Federal student loan balance, Virginia | $46.9B | $23.9B (-49.1%) | $37.9B (-19.2%) | $46.9B (-0.0%) | $31.0B (-34.0%) | $46.9B (-0.0%) |
| Federal student loan balance, Washington | $30.5B | $16.0B (-47.4%) | $13.4B (-56.1%) | $30.6B (+0.2%) | $15.4B (-49.5%) | $30.5B (-0.0%) |
| Federal student loan balance, West Virginia | $7.9B | $5.6B (-29.1%) | $6.4B (-19.2%) | $7.9B (-0.0%) | $5.6B (-29.9%) | $8.0B (+0.1%) |
| Federal student loan balance, Wisconsin | $24.8B | $11.6B (-53.1%) | $11.8B (-52.5%) | $24.7B (-0.1%) | $11.0B (-55.5%) | $24.7B (-0.2%) |
| Federal student loan balance, Wyoming | $1.9B | $799,653,011 (-56.9%) | $494,870,182 (-73.4%) | $1.9B (-0.3%) | $732,630,963 (-60.6%) | $1.9B (-0.2%) |
| Federal student loan borrowers, Alabama | 694,297 | 299,800 (-56.8%) | 254,906 (-63.3%) | 694,484 (+0.0%) | 299,798 (-56.8%) | 695,397 (+0.2%) |
| Federal student loan borrowers, Alaska | 67,443 | 42,367 (-37.2%) | 43,515 (-35.5%) | 67,419 (-0.0%) | 41,980 (-37.8%) | 67,404 (-0.1%) |
| Federal student loan borrowers, Arizona | 957,900 | 568,336 (-40.7%) | 475,180 (-50.4%) | 957,726 (-0.0%) | 539,307 (-43.7%) | 956,166 (-0.2%) |
| Federal student loan borrowers, Arkansas | 427,034 | 179,315 (-58.0%) | 166,818 (-60.9%) | 427,506 (+0.1%) | 181,962 (-57.4%) | 377,427 (-11.6%) |
| Federal student loan borrowers, California | 4.10M | 2.20M (-46.4%) | 2.31M (-43.7%) | 4.10M (+0.0%) | 2.16M (-47.3%) | 4.10M (+0.0%) |
| Federal student loan borrowers, Colorado | 799,487 | 376,661 (-52.9%) | 287,576 (-64.0%) | 800,730 (+0.2%) | 307,585 (-61.5%) | 797,681 (-0.2%) |
| Federal student loan borrowers, Connecticut | 538,917 | 200,218 (-62.8%) | 215,616 (-60.0%) | 428,970 (-20.4%) | 214,014 (-60.3%) | 357,504 (-33.7%) |
| Federal student loan borrowers, Delaware | 144,088 | 61,556 (-57.3%) | 66,317 (-54.0%) | 120,920 (-16.1%) | 62,407 (-56.7%) | 109,128 (-24.3%) |
| Federal student loan borrowers, District of Columbia | 119,829 | 56,416 (-52.9%) | 45,141 (-62.3%) | 119,785 (-0.0%) | 54,249 (-54.7%) | 104,429 (-12.9%) |
| Federal student loan borrowers, Florida | 2.87M | 1.02M (-64.6%) | 1.18M (-59.0%) | 2.87M (-0.0%) | 1.07M (-62.8%) | 2.79M (-2.9%) |
| Federal student loan borrowers, Georgia | 1.78M | 895,114 (-49.7%) | 949,522 (-46.7%) | 1.78M (-0.0%) | 803,541 (-54.9%) | 1.72M (-3.3%) |
| Federal student loan borrowers, Hawaii | 126,521 | 70,947 (-43.9%) | 79,600 (-37.1%) | 126,442 (-0.1%) | 66,457 (-47.5%) | 114,039 (-9.9%) |
| Federal student loan borrowers, Idaho | 226,378 | 158,278 (-30.1%) | 154,004 (-32.0%) | 226,432 (+0.0%) | 163,950 (-27.6%) | 225,793 (-0.3%) |
| Federal student loan borrowers, Illinois | 1.67M | 809,565 (-51.7%) | 991,286 (-40.8%) | 1.67M (-0.1%) | 949,750 (-43.3%) | 1.67M (-0.1%) |
| Federal student loan borrowers, Indiana | 939,497 | 566,356 (-39.7%) | 576,460 (-38.6%) | 939,405 (-0.0%) | 606,899 (-35.4%) | 939,251 (-0.0%) |
| Federal student loan borrowers, Iowa | 442,719 | 197,218 (-55.5%) | 239,760 (-45.8%) | 442,894 (+0.0%) | 237,122 (-46.4%) | 440,443 (-0.5%) |
| Federal student loan borrowers, Kansas | 396,607 | 227,692 (-42.6%) | 222,096 (-44.0%) | 396,315 (-0.1%) | 214,203 (-46.0%) | 396,666 (+0.0%) |
| Federal student loan borrowers, Kentucky | 643,689 | 292,355 (-54.6%) | 324,504 (-49.6%) | 643,650 (-0.0%) | 301,880 (-53.1%) | 494,346 (-23.2%) |
| Federal student loan borrowers, Louisiana | 725,666 | 221,291 (-69.5%) | 259,676 (-64.2%) | 725,630 (-0.0%) | 233,994 (-67.8%) | 608,283 (-16.2%) |
| Federal student loan borrowers, Maine | 195,114 | 116,910 (-40.1%) | 93,095 (-52.3%) | 163,309 (-16.3%) | 108,179 (-44.6%) | 152,248 (-22.0%) |
| Federal student loan borrowers, Maryland | 880,732 | 365,740 (-58.5%) | 361,944 (-58.9%) | 772,806 (-12.3%) | 333,874 (-62.1%) | 653,671 (-25.8%) |
| Federal student loan borrowers, Massachusetts | 941,693 | 468,114 (-50.3%) | 410,147 (-56.4%) | 942,709 (+0.1%) | 440,759 (-53.2%) | 927,634 (-1.5%) |
| Federal student loan borrowers, Michigan | 1.44M | 804,634 (-44.1%) | 819,196 (-43.1%) | 1.44M (+0.1%) | 835,678 (-42.0%) | 1.44M (-0.1%) |
| Federal student loan borrowers, Minnesota | 807,852 | 476,572 (-41.0%) | 489,445 (-39.4%) | 806,823 (-0.1%) | 499,984 (-38.1%) | 769,185 (-4.8%) |
| Federal student loan borrowers, Mississippi | 477,120 | 212,627 (-55.4%) | 199,676 (-58.1%) | 476,743 (-0.1%) | 225,931 (-52.6%) | 446,337 (-6.5%) |
| Federal student loan borrowers, Missouri | 864,211 | 452,444 (-47.6%) | 553,444 (-36.0%) | 865,514 (+0.2%) | 473,298 (-45.2%) | 769,360 (-11.0%) |
| Federal student loan borrowers, Montana | 132,167 | 93,724 (-29.1%) | 86,614 (-34.5%) | 132,178 (+0.0%) | 94,461 (-28.5%) | 131,897 (-0.2%) |
| Federal student loan borrowers, Nebraska | 254,610 | 111,991 (-56.0%) | 120,989 (-52.5%) | 255,006 (+0.2%) | 105,848 (-58.4%) | 210,699 (-17.2%) |
| Federal student loan borrowers, Nevada | 384,687 | 177,125 (-54.0%) | 213,697 (-44.4%) | 384,138 (-0.1%) | 196,577 (-48.9%) | 354,481 (-7.9%) |
| Federal student loan borrowers, New Hampshire | 196,264 | 108,023 (-45.0%) | 103,378 (-47.3%) | 196,628 (+0.2%) | 109,233 (-44.3%) | 196,426 (+0.1%) |
| Federal student loan borrowers, New Jersey | 1.29M | 447,680 (-65.3%) | 536,510 (-58.5%) | 1.29M (+0.1%) | 483,073 (-62.6%) | 1.07M (-16.8%) |
| Federal student loan borrowers, New Mexico | 236,312 | 147,762 (-37.5%) | 85,697 (-63.7%) | 235,942 (-0.2%) | 123,887 (-47.6%) | 191,802 (-18.8%) |
| Federal student loan borrowers, New York | 2.57M | 1.24M (-51.8%) | 1.19M (-53.7%) | 2.57M (-0.1%) | 1.28M (-50.3%) | 2.57M (-0.1%) |
| Federal student loan borrowers, North Carolina | 1.44M | 763,659 (-47.1%) | 955,792 (-33.7%) | 1.44M (-0.1%) | 815,913 (-43.4%) | 1.44M (-0.1%) |
| Federal student loan borrowers, North Dakota | 92,434 | 54,709 (-40.8%) | 68,206 (-26.2%) | 92,302 (-0.1%) | 59,660 (-35.5%) | 92,240 (-0.2%) |
| Federal student loan borrowers, Ohio | 1.85M | 706,815 (-61.8%) | 520,187 (-71.9%) | 1.61M (-13.1%) | 624,258 (-66.3%) | 1.44M (-22.4%) |
| Federal student loan borrowers, Oklahoma | 534,002 | 320,839 (-39.9%) | 276,825 (-48.2%) | 533,733 (-0.1%) | 296,474 (-44.5%) | 532,759 (-0.2%) |
| Federal student loan borrowers, Oregon | 550,628 | 297,551 (-46.0%) | 322,529 (-41.4%) | 550,760 (+0.0%) | 320,994 (-41.7%) | 551,860 (+0.2%) |
| Federal student loan borrowers, Pennsylvania | 1.94M | 698,545 (-64.0%) | 537,921 (-72.3%) | 1.94M (-0.0%) | 646,671 (-66.6%) | 1.93M (-0.3%) |
| Federal student loan borrowers, Rhode Island | 156,531 | 91,440 (-41.6%) | 67,193 (-57.1%) | 156,676 (+0.1%) | 83,389 (-46.7%) | 143,823 (-8.1%) |
| Federal student loan borrowers, South Carolina | 817,995 | 307,285 (-62.4%) | 272,654 (-66.7%) | 817,209 (-0.1%) | 268,422 (-67.2%) | 704,814 (-13.8%) |
| Federal student loan borrowers, South Dakota | 122,025 | 49,229 (-59.7%) | 36,584 (-70.0%) | 122,016 (-0.0%) | 45,954 (-62.3%) | 82,771 (-32.2%) |
| Federal student loan borrowers, Tennessee | 935,732 | 618,646 (-33.9%) | 723,231 (-22.7%) | 935,922 (+0.0%) | 671,009 (-28.3%) | 934,369 (-0.1%) |
| Federal student loan borrowers, Texas | 4.13M | 1.92M (-53.6%) | 1.90M (-54.1%) | 4.13M (-0.1%) | 2.04M (-50.6%) | 4.12M (-0.2%) |
| Federal student loan borrowers, Utah | 327,595 | 229,732 (-29.9%) | 277,395 (-15.3%) | 327,688 (+0.0%) | 274,031 (-16.4%) | 327,472 (-0.0%) |
| Federal student loan borrowers, Vermont | 78,213 | 49,097 (-37.2%) | 49,037 (-37.3%) | 78,112 (-0.1%) | 49,999 (-36.1%) | 78,189 (-0.0%) |
| Federal student loan borrowers, Virginia | 1.14M | 491,787 (-56.7%) | 510,462 (-55.0%) | 1.13M (-0.1%) | 492,105 (-56.6%) | 949,575 (-16.3%) |
| Federal student loan borrowers, Washington | 811,512 | 510,863 (-37.0%) | 398,849 (-50.9%) | 812,134 (+0.1%) | 467,945 (-42.3%) | 810,895 (-0.1%) |
| Federal student loan borrowers, West Virginia | 238,717 | 99,508 (-58.3%) | 111,062 (-53.5%) | 227,134 (-4.9%) | 105,667 (-55.7%) | 160,975 (-32.6%) |
| Federal student loan borrowers, Wisconsin | 741,455 | 324,174 (-56.3%) | 294,341 (-60.3%) | 740,807 (-0.1%) | 318,365 (-57.1%) | 740,101 (-0.2%) |
| Federal student loan borrowers, Wyoming | 57,300 | 35,036 (-38.9%) | 27,385 (-52.2%) | 57,268 (-0.1%) | 31,449 (-45.1%) | 57,300 (-0.0%) |
| Federal student loan balance, borrower age 24 and Younger | $84.9B | $21.8B (-74.3%) | $21.2B (-75.0%) | $84.8B (-0.1%) | $21.3B (-74.9%) | $82.0B (-3.5%) |
| Federal student loan balance, borrower age 25 to 34 | $462.0B | $193.5B (-58.1%) | $192.1B (-58.4%) | $461.4B (-0.1%) | $191.9B (-58.5%) | $454.1B (-1.7%) |
| Federal student loan balance, borrower age 35 to 49 | $679.5B | $347.9B (-48.8%) | $344.7B (-49.3%) | $679.0B (-0.1%) | $341.6B (-49.7%) | $679.6B (+0.0%) |
| Federal student loan balance, borrower age 50 to 61 | $310.2B | $203.8B (-34.3%) | $236.3B (-23.8%) | $310.3B (+0.0%) | $213.6B (-31.1%) | $310.2B (-0.0%) |
| Federal student loan balance, borrower age 62 and Older | $142.5B | $91.4B (-35.9%) | $91.7B (-35.7%) | $142.4B (-0.1%) | $87.3B (-38.7%) | $142.4B (-0.1%) |
| Federal student loan borrowers, borrower age 24 and Younger | 6.13M | 1.06M (-82.7%) | 894,843 (-85.4%) | 3.72M (-39.3%) | 956,370 (-84.4%) | 3.43M (-44.0%) |
| Federal student loan borrowers, borrower age 25 to 34 | 13.85M | 6.19M (-55.3%) | 6.30M (-54.5%) | 14.85M (+7.2%) | 6.26M (-54.8%) | 13.85M (+0.0%) |
| Federal student loan borrowers, borrower age 35 to 49 | 14.84M | 8.02M (-45.9%) | 8.07M (-45.6%) | 15.60M (+5.1%) | 8.19M (-44.8%) | 15.04M (+1.4%) |
| Federal student loan borrowers, borrower age 50 to 61 | 6.33M | 4.41M (-30.3%) | 4.44M (-29.8%) | 6.44M (+1.8%) | 4.46M (-29.5%) | 6.46M (+2.0%) |
| Federal student loan borrowers, borrower age 62 and Older | 3.16M | 1.54M (-51.4%) | 1.74M (-45.0%) | 3.17M (+0.1%) | 1.57M (-50.4%) | 3.17M (+0.0%) |
| Federal student loan balance, borrower debt size Less than 5K | $18.7B | $6.9B (-63.2%) | $6.7B (-64.3%) | $17.0B (-9.1%) | $6.9B (-63.0%) | $14.2B (-24.3%) |
| Federal student loan balance, borrower debt size 5K to 10K | $52.0B | $20.4B (-60.7%) | $18.8B (-63.8%) | $52.0B (+0.0%) | $19.6B (-62.4%) | $43.1B (-17.1%) |
| Federal student loan balance, borrower debt size 10K to 20K | $131.2B | $58.6B (-55.3%) | $60.2B (-54.1%) | $131.1B (-0.0%) | $60.6B (-53.8%) | $131.1B (-0.0%) |
| Federal student loan balance, borrower debt size 20K to 40K | $268.0B | $136.5B (-49.1%) | $147.3B (-45.0%) | $268.2B (+0.1%) | $142.9B (-46.7%) | $267.5B (-0.2%) |
| Federal student loan balance, borrower debt size 40K to 60K | $204.3B | $111.9B (-45.3%) | $102.9B (-49.6%) | $204.4B (+0.0%) | $109.4B (-46.5%) | $204.3B (-0.0%) |
| Federal student loan balance, borrower debt size 60K to 80K | $175.6B | $126.8B (-27.8%) | $142.5B (-18.9%) | $175.7B (+0.1%) | $132.7B (-24.4%) | $178.4B (+1.6%) |
| Federal student loan balance, borrower debt size 80K to 100K | $126.4B | $58.4B (-53.8%) | $53.6B (-57.6%) | $126.4B (-0.0%) | $55.3B (-56.3%) | $126.3B (-0.1%) |
| Federal student loan balance, borrower debt size 100K to 200K | $350.7B | $153.4B (-56.2%) | $159.4B (-54.6%) | $350.9B (+0.0%) | $149.2B (-57.5%) | $350.7B (+0.0%) |
| Federal student loan balance, borrower debt size 200K+ | $352.2B | $185.6B (-47.3%) | $194.7B (-44.7%) | $352.4B (+0.0%) | $179.2B (-49.1%) | $352.6B (+0.1%) |
| Federal student loan borrowers, borrower debt size Less than 5K | 6.82M | 2.66M (-61.0%) | 2.67M (-60.9%) | 6.29M (-7.8%) | 2.75M (-59.7%) | 5.48M (-19.7%) |
| Federal student loan borrowers, borrower debt size 5K to 10K | 7.12M | 2.89M (-59.5%) | 2.66M (-62.7%) | 7.11M (-0.1%) | 2.76M (-61.2%) | 5.94M (-16.5%) |
| Federal student loan borrowers, borrower debt size 10K to 20K | 9.10M | 4.03M (-55.7%) | 4.11M (-54.8%) | 8.85M (-2.8%) | 4.14M (-54.5%) | 8.86M (-2.7%) |
| Federal student loan borrowers, borrower debt size 20K to 40K | 9.40M | 4.99M (-46.9%) | 5.36M (-43.0%) | 9.38M (-0.2%) | 5.21M (-44.5%) | 9.50M (+1.1%) |
| Federal student loan borrowers, borrower debt size 40K to 60K | 4.15M | 2.30M (-44.7%) | 2.12M (-49.0%) | 4.15M (+0.0%) | 2.26M (-45.7%) | 4.15M (+0.0%) |
| Federal student loan borrowers, borrower debt size 60K to 80K | 2.57M | 1.84M (-28.5%) | 2.08M (-19.0%) | 2.55M (-0.7%) | 1.93M (-25.1%) | 2.57M (+0.1%) |
| Federal student loan borrowers, borrower debt size 80K to 100K | 1.38M | 671,311 (-51.5%) | 609,132 (-56.0%) | 1.44M (+3.7%) | 631,351 (-54.4%) | 1.44M (+3.7%) |
| Federal student loan borrowers, borrower debt size 100K to 200K | 2.57M | 1.19M (-53.7%) | 1.20M (-53.5%) | 2.63M (+2.2%) | 1.15M (-55.4%) | 2.65M (+3.2%) |
| Federal student loan borrowers, borrower debt size 200K+ | 1.19M | 656,753 (-44.7%) | 642,242 (-45.9%) | 1.37M (+15.7%) | 609,324 (-48.7%) | 1.35M (+13.8%) |
| Parent PLUS loan balance | $114.8B | $143.9B (+25.3%) | $152.6B (+32.8%) | $114.9B (+0.0%) | $143.8B (+25.2%) | $126.7B (+10.3%) |
| Federal student loan balance excluding Parent PLUS | $1,564.2B | $714.6B (-54.3%) | $733.5B (-53.1%) | $1,563.1B (-0.1%) | $712.0B (-54.5%) | $1,541.5B (-1.5%) |
