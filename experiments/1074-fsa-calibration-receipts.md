# microcosm#1074 receipts: Federal Student Aid portfolio targets

Generated 2026-10-23 by `experiments/fit_1074_fsa_calibration.py`, commit `4b6bfce9a240cd44c45210fbf508640d8402343d`, Modal run `20261023T191527Z-4b6bfce9a` (`experiments/modal_1074.py`). Machine-readable copy: `experiments/receipts/1074-fsa-calibration-fit.json`.

Host: the published default `populace-us-2024-spm-20260915` (sha256 `6496cc4393d4…`, 57,240 households), with the SCF 2025 education-debt stage run at seed 0. Feed: the pinned `51aa40f` US feed plus the merged chronicle#347 FSA rows (`d8e3d0fd9e21…`, 40,639 rows). Arms: **release** = the published weights; **A** = a dense refit to the `national_state` surface without the FSA family; **B** = the same refit with it. Same epochs (1500), learning rate, 5x weight cap and conserved mass as a release. Not a release: no gate ran and nothing was published.

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
| fsa_age_band_balance | 5 | 47.5% (0%) | 45.0% (0%) | 0.1% (100%) | 47.1% (0%) | 0.3% (100%) |
| fsa_age_band_borrowers | 5 | 54.5% (0%) | 53.7% (0%) | 0.2% (100%) | 54.1% (0%) | 6.5% (40%) |
| fsa_balance_excluding_parent_plus | 1 | 51.0% (0%) | 49.9% (0%) | 0.1% (100%) | 51.2% (0%) | 0.5% (100%) |
| fsa_debt_size_band_balance | 9 | 50.6% (0%) | 49.8% (0%) | 0.4% (100%) | 50.5% (0%) | 3.6% (78%) |
| fsa_debt_size_band_borrowers | 9 | 49.0% (0%) | 49.0% (0%) | 2.7% (89%) | 49.3% (0%) | 5.3% (67%) |
| fsa_parent_plus_balance | 1 | 20.1% (0%) | 10.8% (0%) | 0.0% (100%) | 18.9% (0%) | 0.0% (100%) |
| fsa_state_balance | 51 | 47.0% (0%) | 46.7% (2%) | 0.2% (98%) | 47.0% (0%) | 1.0% (96%) |
| fsa_state_borrowers | 51 | 49.1% (0%) | 49.4% (0%) | 1.1% (92%) | 48.9% (0%) | 6.2% (63%) |

## Headline levels

| | FSA target | Release | A, no FSA | B, with FSA | A + L2 | B + L2 |
|---|---:|---:|---:|---:|---:|---:|
| Federal balance, own plus Parent PLUS | $1,679.1B | $858.4B | $886.0B | $1,677.3B | $855.8B | $1,671.2B |
| Borrowers (people with a balance) | 44.31M | 21.18M | 21.59M | 44.26M | 21.42M | 42.67M |
| Parent PLUS balance | $114.8B | $91.8B | $102.5B | $114.8B | $93.2B | $114.8B |
| Parent PLUS share of the balance | 6.8% | 10.7% | 11.6% | 6.8% | 10.9% | 6.9% |
| Mean balance per borrower | $37,894 | $40,535 | $41,040 | $37,897 | $39,951 | $39,162 |
| Households with federal education debt |  | 16.4% | 16.6% | 33.7% | 16.6% | 32.3% |

## Every other family: does adding FSA cost the rest of the surface?

Mean absolute relative error over the `national_state` surface's non-FSA targets, by provider/category (the dashboard's family), each miss capped at 100% of its target as the loss caps it, sorted by how much adding the FSA family moves it:

| Family | Targets | Release | A, no FSA | B, with FSA | A + L2 | B + L2 | B - A (pp) | B+L2 - A+L2 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hhs_acf_tanf/hhs_acf_tanf.hhs_acf_tanf | 30 | 29.8% | 18.3% | 20.9% | 22.5% | 22.4% | +2.56 | -0.09 |
| ssa/ssa.ssi_federal_payment_recipients | 3 | 28.5% | 15.6% | 17.5% | 24.3% | 23.1% | +1.83 | -1.25 |
| jct/jct.individual_tax_expenditure_revenue_loss | 11 | 42.3% | 35.3% | 37.1% | 39.9% | 39.5% | +1.72 | -0.39 |
| bea/bea.nipa_proprietors_income | 1 | 11.6% | 5.9% | 7.3% | 11.2% | 9.6% | +1.40 | -1.64 |
| irs_soi/irs_soi.capital_gains_gross | 104 | 30.1% | 17.0% | 18.1% | 26.0% | 24.8% | +1.12 | -1.18 |
| irs_soi/irs_soi.real_estate_taxes | 106 | 6.5% | 5.3% | 6.3% | 6.3% | 6.6% | +1.00 | +0.29 |
| irs_soi/irs_soi.assigned_aca_ptc | 104 | 0.9% | 0.6% | 1.5% | 0.7% | 1.1% | +0.91 | +0.37 |
| irs_soi/irs_soi.refundable_ctc | 104 | 0.8% | 0.3% | 1.0% | 0.7% | 1.1% | +0.70 | +0.41 |
| irs_soi/irs_soi.eitc | 781 | 3.1% | 2.5% | 3.1% | 2.7% | 3.1% | +0.67 | +0.39 |
| irs_soi/irs_soi.ctc | 104 | 1.5% | 0.6% | 1.2% | 1.2% | 1.4% | +0.59 | +0.22 |
| irs_soi/irs_soi.salt_deduction | 106 | 1.9% | 1.8% | 2.4% | 2.1% | 2.1% | +0.58 | +0.05 |
| irs_soi/irs_soi.charitable_deduction | 2 | 7.5% | 6.8% | 7.3% | 6.2% | 6.6% | +0.49 | +0.45 |
| irs_soi/irs_soi.interest_deduction | 2 | 2.0% | 1.7% | 1.3% | 2.4% | 2.0% | -0.46 | -0.39 |
| irs_soi/irs_soi.taxable_interest_income | 1038 | 6.0% | 5.7% | 6.2% | 6.0% | 5.9% | +0.45 | -0.05 |
| irs_soi/irs_soi.count | 260 | 31.7% | 20.5% | 21.0% | 22.9% | 22.3% | +0.45 | -0.58 |
| irs_soi/irs_soi.unemployment_compensation | 106 | 0.1% | 0.3% | 0.5% | 0.2% | 0.6% | +0.27 | +0.41 |
| census_stc/census_stc.individual_income_tax_collections | 44 | 13.9% | 13.7% | 14.0% | 13.9% | 14.0% | +0.26 | +0.06 |
| irs_soi/irs_soi.qualified_dividends | 104 | 0.3% | 0.5% | 0.7% | 0.5% | 0.5% | +0.19 | +0.02 |
| irs_soi/irs_soi.income_tax | 104 | 0.6% | 0.7% | 0.8% | 0.8% | 0.9% | +0.14 | +0.09 |
| irs_soi/irs_soi.ordinary_dividend_income | 108 | 0.8% | 0.8% | 0.9% | 0.8% | 0.8% | +0.14 | -0.08 |
| irs_soi/irs_soi.tax_exempt_interest_income | 104 | 0.1% | 0.7% | 0.8% | 1.0% | 1.0% | +0.13 | -0.02 |
| irs_soi/irs_soi.partnership_and_s_corp_income | 104 | 0.6% | 0.4% | 0.5% | 0.6% | 0.6% | +0.11 | -0.05 |
| irs_soi/irs_soi.taxable_income | 107 | 0.3% | 0.4% | 0.5% | 0.6% | 0.6% | +0.10 | +0.02 |
| irs_soi/irs_soi.tip_income | 1 | 21.3% | 0.1% | 0.0% | 0.1% | 0.0% | -0.09 | -0.03 |
| cms_medicare/cms_medicare.cms_medicare | 1 | 13.8% | 13.5% | 13.6% | 13.5% | 13.4% | +0.09 | -0.05 |
| irs_soi/irs_soi.income_tax_before_credits | 104 | 1.6% | 1.6% | 1.7% | 1.9% | 2.0% | +0.09 | +0.07 |
| irs_soi/irs_soi.estate_losses | 2 | 50.0% | 50.2% | 50.1% | 50.2% | 50.0% | -0.08 | -0.18 |
| irs_soi/irs_soi.medical_expense_deduction | 104 | 6.0% | 6.3% | 6.3% | 6.2% | 6.5% | +0.08 | +0.26 |
| ssa/ssa.ssi_recipients | 51 | 4.8% | 0.7% | 0.8% | 2.0% | 1.5% | +0.08 | -0.47 |
| ssa/ssa.ssi_payments | 51 | 3.7% | 0.4% | 0.5% | 0.5% | 1.0% | +0.07 | +0.45 |
| bea/bea.nipa_wages_and_salaries | 1 | 13.3% | 13.6% | 13.6% | 13.5% | 13.5% | -0.07 | +0.06 |
| irs_soi/irs_soi.adjusted_gross_income | 261 | 19.7% | 0.6% | 0.6% | 2.8% | 2.2% | +0.07 | -0.59 |
| usda_snap/usda_snap.usda_snap | 104 | 7.9% | 0.1% | 0.2% | 0.6% | 0.7% | +0.05 | +0.07 |
| irs_soi/irs_soi.employment_income | 106 | 0.1% | 0.1% | 0.2% | 0.2% | 0.2% | +0.05 | +0.02 |
| irs_soi/irs_soi.rent_and_royalty_net_income | 104 | 4.0% | 3.1% | 3.1% | 3.1% | 3.1% | +0.04 | -0.03 |
| irs_soi/irs_soi.estate_income | 2 | 0.0% | 0.1% | 0.1% | 0.1% | 0.1% | -0.03 | +0.01 |
| ssa/ssa.ssa | 6 | 3.7% | 0.1% | 0.1% | 1.4% | 1.3% | -0.02 | -0.10 |
| cms_medicaid/cms_medicaid.cms_medicaid | 135 | 3.1% | 2.3% | 2.3% | 2.3% | 2.3% | +0.02 | -0.00 |
| irs_soi/irs_soi.ira_distributions | 108 | 0.1% | 0.1% | 0.2% | 0.1% | 0.1% | +0.02 | +0.00 |
| irs_soi/irs_soi.non_sch_d_capital_gains | 2 | 0.0% | 0.0% | 0.1% | 0.1% | 0.0% | +0.02 | -0.06 |
| irs_soi/irs_soi.deductible_mortgage_interest | 2 | 0.4% | 0.0% | 0.0% | 0.1% | 0.0% | -0.01 | -0.07 |
| federal_reserve/federal_reserve.federal_reserve | 1 | 0.0% | 0.0% | 0.0% | 0.1% | 0.0% | -0.01 | -0.04 |
| irs_soi/irs_soi.business_net_profits | 104 | 1.3% | 1.1% | 1.1% | 1.1% | 1.1% | -0.01 | +0.02 |
| irs_soi/irs_soi.taxable_social_security | 108 | 0.3% | 0.2% | 0.3% | 0.3% | 0.3% | +0.01 | +0.01 |
| cbo/cbo.cbo | 5 | 4.5% | 4.6% | 4.6% | 4.6% | 4.6% | +0.01 | +0.07 |
| cms_aca/cms_aca.cms_aca | 102 | 1.6% | 0.2% | 0.2% | 0.7% | 0.6% | +0.01 | -0.12 |
| census_pep/census_pep.resident_population | 936 | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | +0.00 | +0.01 |
| hhs_acf_liheap/hhs_acf_liheap.hhs_acf_liheap | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | -0.00 | +0.01 |
| irs_soi/irs_soi.taxable_pension_income | 108 | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | -0.00 | +0.00 |
| irs_soi/irs_soi.itemized_taxable_income_deductions | 55 | 2.0% | 2.3% | 2.3% | 2.5% | 2.4% | -0.00 | -0.16 |
| **all non-FSA targets** | 6102 | 5.3% | 3.3% | 3.6% | 3.8% | 3.9% | +0.31 | +0.00 |

## Racial composition of borrowers, before and after calibration

Borrower households by the reference person's race (the SCF's `racecl4`; the SCF column pools the five implicates):

| | SCF 2025 | Release | A, no FSA | B, with FSA | A + L2 | B + L2 | B - release (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| white_non_hispanic | 57.6% | 60.8% | 60.0% | 57.6% | 62.1% | 59.9% | -3.2 |
| black_non_hispanic | 18.6% | 17.4% | 16.4% | 18.8% | 16.8% | 19.2% | +1.5 |
| hispanic | 9.2% | 12.9% | 13.8% | 14.4% | 12.4% | 12.4% | +1.5 |
| other_or_multiple | 14.7% | 9.0% | 9.8% | 9.2% | 8.7% | 8.5% | +0.2 |

Borrowers (people) by their own race, and the share of adults who borrow:

| | Release | A, no FSA | B, with FSA | A + L2 | B + L2 | Adults with a balance: release | B | B + L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| white_non_hispanic | 60.2% | 59.3% | 57.2% | 61.4% | 59.4% | 7.7% | 15.7% | 14.9% |
| black_non_hispanic | 16.6% | 16.2% | 18.4% | 16.4% | 18.4% | 12.9% | 26.8% | 26.5% |
| hispanic | 13.5% | 13.9% | 14.7% | 12.6% | 12.7% | 6.4% | 14.8% | 14.6% |
| other_or_multiple | 9.6% | 10.6% | 9.7% | 9.6% | 9.4% | 8.1% | 15.4% | 15.5% |

## Weight movement against the published weights

| | A, no FSA | B, with FSA | A + L2 | B + L2 |
|---|---:|---:|---:|---:|
| Borrower households' weight ratio | 1.01 | 2.05 | 1.01 | 1.97 |
| Other households' weight ratio | 0.998 | 0.793 | 0.998 | 0.810 |
| Borrower households' share of weight | 16.6% | 33.7% | 16.6% | 32.3% |
| Median household weight ratio | 0.02 | 7.0e-04 | 0.13 | 0.01 |
| 99th percentile weight ratio | 4.76 | 4.98 | 1.97 | 2.77 |
| Effective sample size (release 9,174) | 4,149 | 3,045 | 6,435 | 4,424 |

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
| Federal student loan balance, Alabama | $26.8B | $8.3B (-69.0%) | $7.3B (-72.7%) | $26.8B (-0.1%) | $7.9B (-70.7%) | $26.4B (-1.8%) |
| Federal student loan balance, Alaska | $2.5B | $1.1B (-53.6%) | $1.4B (-42.4%) | $2.5B (-0.2%) | $1.2B (-52.7%) | $2.2B (-11.4%) |
| Federal student loan balance, Arizona | $35.1B | $24.3B (-30.7%) | $14.4B (-59.0%) | $35.1B (-0.0%) | $18.5B (-47.1%) | $35.0B (-0.2%) |
| Federal student loan balance, Arkansas | $15.0B | $6.0B (-59.9%) | $6.1B (-59.3%) | $15.0B (-0.0%) | $6.3B (-57.6%) | $15.0B (+0.0%) |
| Federal student loan balance, California | $161.7B | $91.3B (-43.5%) | $99.2B (-38.6%) | $161.8B (+0.1%) | $96.8B (-40.1%) | $161.4B (-0.2%) |
| Federal student loan balance, Colorado | $30.6B | $12.2B (-60.0%) | $6.4B (-79.1%) | $30.6B (-0.2%) | $9.0B (-70.5%) | $30.6B (-0.2%) |
| Federal student loan balance, Connecticut | $20.4B | $8.2B (-59.8%) | $8.1B (-60.2%) | $20.5B (+0.1%) | $7.9B (-61.1%) | $20.3B (-0.5%) |
| Federal student loan balance, Delaware | $5.8B | $2.6B (-54.7%) | $3.3B (-42.6%) | $5.8B (+0.1%) | $2.9B (-49.4%) | $5.5B (-3.9%) |
| Federal student loan balance, District of Columbia | $6.6B | $2.6B (-60.6%) | $1.5B (-77.5%) | $6.6B (+0.1%) | $2.5B (-62.1%) | $6.6B (-0.2%) |
| Federal student loan balance, Florida | $116.7B | $42.9B (-63.2%) | $43.9B (-62.4%) | $116.7B (+0.0%) | $42.8B (-63.4%) | $116.7B (+0.0%) |
| Federal student loan balance, Georgia | $77.0B | $38.4B (-50.1%) | $51.9B (-32.5%) | $77.0B (+0.0%) | $33.3B (-56.8%) | $76.9B (-0.1%) |
| Federal student loan balance, Hawaii | $5.1B | $3.3B (-35.6%) | $3.2B (-36.0%) | $5.0B (-0.2%) | $3.2B (-36.7%) | $5.1B (+0.1%) |
| Federal student loan balance, Idaho | $7.7B | $6.2B (-20.5%) | $6.4B (-17.2%) | $7.7B (-0.1%) | $6.5B (-16.0%) | $7.7B (-0.2%) |
| Federal student loan balance, Illinois | $67.4B | $34.1B (-49.3%) | $50.4B (-25.2%) | $67.2B (-0.3%) | $47.7B (-29.2%) | $67.5B (+0.2%) |
| Federal student loan balance, Indiana | $32.0B | $25.3B (-20.8%) | $26.4B (-17.3%) | $32.0B (+0.1%) | $28.2B (-11.8%) | $32.0B (+0.1%) |
| Federal student loan balance, Iowa | $13.9B | $5.5B (-60.6%) | $6.5B (-53.1%) | $13.9B (+0.0%) | $6.1B (-56.5%) | $13.9B (-0.1%) |
| Federal student loan balance, Kansas | $13.5B | $6.5B (-52.1%) | $6.5B (-52.1%) | $13.5B (+0.2%) | $6.9B (-49.2%) | $13.5B (-0.2%) |
| Federal student loan balance, Kentucky | $22.3B | $12.8B (-42.6%) | $20.2B (-9.5%) | $22.3B (+0.1%) | $13.2B (-40.6%) | $22.3B (+0.1%) |
| Federal student loan balance, Louisiana | $25.9B | $6.1B (-76.6%) | $9.0B (-65.4%) | $24.0B (-7.3%) | $6.6B (-74.4%) | $20.4B (-21.4%) |
| Federal student loan balance, Maine | $6.9B | $4.0B (-42.0%) | $2.7B (-60.9%) | $6.9B (-0.1%) | $3.6B (-48.5%) | $6.9B (-0.2%) |
| Federal student loan balance, Maryland | $39.6B | $14.4B (-63.8%) | $15.2B (-61.5%) | $39.6B (+0.0%) | $14.4B (-63.6%) | $39.6B (+0.0%) |
| Federal student loan balance, Massachusetts | $34.5B | $19.3B (-44.0%) | $16.5B (-52.3%) | $34.4B (-0.0%) | $18.7B (-45.8%) | $34.4B (-0.0%) |
| Federal student loan balance, Michigan | $54.9B | $34.0B (-38.1%) | $39.5B (-28.0%) | $54.9B (+0.0%) | $34.5B (-37.1%) | $54.9B (+0.0%) |
| Federal student loan balance, Minnesota | $28.4B | $20.9B (-26.4%) | $21.5B (-24.1%) | $28.4B (+0.1%) | $22.3B (-21.3%) | $28.4B (+0.0%) |
| Federal student loan balance, Mississippi | $18.4B | $7.9B (-57.1%) | $6.3B (-65.8%) | $18.4B (+0.0%) | $7.9B (-56.9%) | $18.4B (+0.0%) |
| Federal student loan balance, Missouri | $31.6B | $15.7B (-50.2%) | $24.7B (-21.7%) | $31.6B (+0.1%) | $18.4B (-41.7%) | $31.6B (+0.2%) |
| Federal student loan balance, Montana | $4.6B | $3.6B (-23.5%) | $3.5B (-25.1%) | $4.7B (+0.2%) | $3.6B (-22.8%) | $4.6B (-0.1%) |
| Federal student loan balance, Nebraska | $8.5B | $3.9B (-54.3%) | $4.3B (-49.1%) | $8.5B (-0.1%) | $3.7B (-56.7%) | $8.5B (-0.0%) |
| Federal student loan balance, Nevada | $13.6B | $5.4B (-60.4%) | $6.8B (-50.2%) | $13.6B (-0.1%) | $6.0B (-55.9%) | $13.4B (-1.5%) |
| Federal student loan balance, New Hampshire | $7.0B | $3.6B (-48.4%) | $2.9B (-58.9%) | $7.0B (-0.0%) | $3.1B (-55.9%) | $7.0B (-0.1%) |
| Federal student loan balance, New Jersey | $49.9B | $22.9B (-54.1%) | $27.5B (-44.9%) | $49.9B (+0.0%) | $24.2B (-51.5%) | $50.0B (+0.2%) |
| Federal student loan balance, New Mexico | $8.3B | $9.6B (+15.8%) | $4.6B (-44.1%) | $8.3B (+0.1%) | $7.4B (-10.5%) | $8.3B (+0.1%) |
| Federal student loan balance, New York | $103.1B | $59.0B (-42.7%) | $52.1B (-49.4%) | $103.0B (-0.1%) | $54.5B (-47.1%) | $103.3B (+0.2%) |
| Federal student loan balance, North Carolina | $57.6B | $30.7B (-46.7%) | $48.5B (-15.8%) | $57.7B (+0.2%) | $34.4B (-40.2%) | $57.6B (+0.1%) |
| Federal student loan balance, North Dakota | $2.8B | $1.5B (-47.1%) | $1.6B (-42.6%) | $2.8B (+0.1%) | $1.6B (-44.3%) | $2.8B (-0.4%) |
| Federal student loan balance, Ohio | $66.3B | $24.9B (-62.4%) | $17.5B (-73.7%) | $66.3B (-0.0%) | $22.7B (-65.7%) | $66.2B (-0.2%) |
| Federal student loan balance, Oklahoma | $17.6B | $12.7B (-28.1%) | $10.4B (-41.1%) | $17.7B (+0.2%) | $11.9B (-32.6%) | $17.7B (+0.2%) |
| Federal student loan balance, Oregon | $21.5B | $9.8B (-54.5%) | $10.2B (-52.6%) | $21.4B (-0.1%) | $9.8B (-54.5%) | $20.9B (-2.7%) |
| Federal student loan balance, Pennsylvania | $71.8B | $27.9B (-61.1%) | $19.5B (-72.9%) | $71.9B (+0.1%) | $24.6B (-65.7%) | $71.9B (+0.2%) |
| Federal student loan balance, Rhode Island | $5.3B | $2.8B (-46.9%) | $2.2B (-58.6%) | $5.3B (-0.1%) | $2.7B (-48.8%) | $5.3B (-0.2%) |
| Federal student loan balance, South Carolina | $32.4B | $17.5B (-46.1%) | $14.4B (-55.5%) | $32.4B (+0.1%) | $13.3B (-58.9%) | $32.4B (-0.1%) |
| Federal student loan balance, South Dakota | $3.8B | $2.5B (-33.3%) | $1.2B (-67.7%) | $3.8B (-0.2%) | $1.8B (-52.5%) | $3.8B (+0.0%) |
| Federal student loan balance, Tennessee | $35.7B | $30.7B (-14.0%) | $34.8B (-2.6%) | $35.7B (-0.0%) | $30.9B (-13.5%) | $35.7B (-0.1%) |
| Federal student loan balance, Texas | $142.7B | $69.0B (-51.6%) | $44.8B (-68.6%) | $142.6B (-0.0%) | $58.3B (-59.2%) | $142.3B (-0.2%) |
| Federal student loan balance, Utah | $11.5B | $7.1B (-38.2%) | $9.2B (-19.8%) | $11.4B (-0.0%) | $8.6B (-24.6%) | $11.5B (+0.0%) |
| Federal student loan balance, Vermont | $3.0B | $1.5B (-49.1%) | $1.4B (-51.9%) | $3.0B (-0.2%) | $1.6B (-48.0%) | $3.0B (-0.2%) |
| Federal student loan balance, Virginia | $46.9B | $23.9B (-49.1%) | $37.9B (-19.2%) | $46.9B (-0.2%) | $31.0B (-34.0%) | $47.0B (+0.0%) |
| Federal student loan balance, Washington | $30.5B | $16.0B (-47.4%) | $13.4B (-56.1%) | $30.6B (+0.1%) | $15.4B (-49.5%) | $30.4B (-0.4%) |
| Federal student loan balance, West Virginia | $7.9B | $5.6B (-29.1%) | $6.4B (-19.2%) | $7.9B (-0.1%) | $5.6B (-29.9%) | $7.9B (-0.1%) |
| Federal student loan balance, Wisconsin | $24.8B | $11.6B (-53.1%) | $11.8B (-52.5%) | $24.8B (+0.0%) | $11.0B (-55.5%) | $24.7B (-0.1%) |
| Federal student loan balance, Wyoming | $1.9B | $799,653,011 (-56.9%) | $494,870,182 (-73.4%) | $1.9B (+0.2%) | $732,630,963 (-60.6%) | $1.9B (+0.2%) |
| Federal student loan borrowers, Alabama | 694,297 | 299,661 (-56.8%) | 250,982 (-63.9%) | 693,709 (-0.1%) | 306,411 (-55.9%) | 693,173 (-0.2%) |
| Federal student loan borrowers, Alaska | 67,443 | 42,678 (-36.7%) | 43,211 (-35.9%) | 67,354 (-0.1%) | 41,674 (-38.2%) | 67,485 (+0.1%) |
| Federal student loan borrowers, Arizona | 957,900 | 581,654 (-39.3%) | 478,956 (-50.0%) | 956,836 (-0.1%) | 553,297 (-42.2%) | 957,195 (-0.1%) |
| Federal student loan borrowers, Arkansas | 427,034 | 178,289 (-58.2%) | 163,756 (-61.7%) | 426,375 (-0.2%) | 178,797 (-58.1%) | 380,809 (-10.8%) |
| Federal student loan borrowers, California | 4.10M | 2.28M (-44.4%) | 2.44M (-40.5%) | 4.23M (+3.3%) | 2.28M (-44.4%) | 4.10M (+0.1%) |
| Federal student loan borrowers, Colorado | 799,487 | 334,926 (-58.1%) | 239,763 (-70.0%) | 798,939 (-0.1%) | 263,859 (-67.0%) | 750,791 (-6.1%) |
| Federal student loan borrowers, Connecticut | 538,917 | 209,988 (-61.0%) | 227,777 (-57.7%) | 483,745 (-10.2%) | 221,602 (-58.9%) | 389,148 (-27.8%) |
| Federal student loan borrowers, Delaware | 144,088 | 59,672 (-58.6%) | 65,530 (-54.5%) | 125,864 (-12.6%) | 61,575 (-57.3%) | 113,234 (-21.4%) |
| Federal student loan borrowers, District of Columbia | 119,829 | 54,421 (-54.6%) | 42,781 (-64.3%) | 119,567 (-0.2%) | 52,682 (-56.0%) | 98,909 (-17.5%) |
| Federal student loan borrowers, Florida | 2.87M | 987,744 (-65.6%) | 1.16M (-59.5%) | 2.87M (-0.0%) | 1.05M (-63.6%) | 2.87M (-0.2%) |
| Federal student loan borrowers, Georgia | 1.78M | 917,964 (-48.4%) | 963,914 (-45.9%) | 1.78M (+0.0%) | 823,647 (-53.7%) | 1.78M (-0.0%) |
| Federal student loan borrowers, Hawaii | 126,521 | 67,915 (-46.3%) | 69,764 (-44.9%) | 126,286 (-0.2%) | 64,639 (-48.9%) | 108,157 (-14.5%) |
| Federal student loan borrowers, Idaho | 226,378 | 167,325 (-26.1%) | 169,172 (-25.3%) | 226,231 (-0.1%) | 181,780 (-19.7%) | 226,158 (-0.1%) |
| Federal student loan borrowers, Illinois | 1.67M | 832,380 (-50.3%) | 1.02M (-39.0%) | 1.67M (-0.1%) | 966,614 (-42.3%) | 1.68M (+0.1%) |
| Federal student loan borrowers, Indiana | 939,497 | 516,324 (-45.0%) | 518,545 (-44.8%) | 939,741 (+0.0%) | 554,968 (-40.9%) | 921,590 (-1.9%) |
| Federal student loan borrowers, Iowa | 442,719 | 212,727 (-51.9%) | 250,949 (-43.3%) | 442,914 (+0.0%) | 250,321 (-43.5%) | 443,024 (+0.1%) |
| Federal student loan borrowers, Kansas | 396,607 | 215,828 (-45.6%) | 218,049 (-45.0%) | 397,536 (+0.2%) | 208,432 (-47.4%) | 396,131 (-0.1%) |
| Federal student loan borrowers, Kentucky | 643,689 | 292,355 (-54.6%) | 324,504 (-49.6%) | 612,160 (-4.9%) | 301,880 (-53.1%) | 497,613 (-22.7%) |
| Federal student loan borrowers, Louisiana | 725,666 | 235,996 (-67.5%) | 283,225 (-61.0%) | 726,299 (+0.1%) | 256,063 (-64.7%) | 713,961 (-1.6%) |
| Federal student loan borrowers, Maine | 195,114 | 118,656 (-39.2%) | 99,175 (-49.2%) | 172,283 (-11.7%) | 111,983 (-42.6%) | 154,282 (-20.9%) |
| Federal student loan borrowers, Maryland | 880,732 | 395,480 (-55.1%) | 371,603 (-57.8%) | 880,547 (-0.0%) | 358,898 (-59.3%) | 848,622 (-3.6%) |
| Federal student loan borrowers, Massachusetts | 941,693 | 463,106 (-50.8%) | 403,617 (-57.1%) | 941,438 (-0.0%) | 436,881 (-53.6%) | 940,063 (-0.2%) |
| Federal student loan borrowers, Michigan | 1.44M | 799,647 (-44.5%) | 832,618 (-42.2%) | 1.44M (+0.1%) | 832,371 (-42.2%) | 1.44M (+0.0%) |
| Federal student loan borrowers, Minnesota | 807,852 | 479,454 (-40.7%) | 503,733 (-37.6%) | 809,206 (+0.2%) | 505,119 (-37.5%) | 806,648 (-0.1%) |
| Federal student loan borrowers, Mississippi | 477,120 | 213,185 (-55.3%) | 208,576 (-56.3%) | 477,629 (+0.1%) | 228,615 (-52.1%) | 476,147 (-0.2%) |
| Federal student loan borrowers, Missouri | 864,211 | 472,395 (-45.3%) | 567,763 (-34.3%) | 865,294 (+0.1%) | 479,060 (-44.6%) | 836,071 (-3.3%) |
| Federal student loan borrowers, Montana | 132,167 | 88,576 (-33.0%) | 81,204 (-38.6%) | 132,441 (+0.2%) | 88,555 (-33.0%) | 129,928 (-1.7%) |
| Federal student loan borrowers, Nebraska | 254,610 | 123,018 (-51.7%) | 144,097 (-43.4%) | 254,409 (-0.1%) | 118,509 (-53.5%) | 246,521 (-3.2%) |
| Federal student loan borrowers, Nevada | 384,687 | 176,512 (-54.1%) | 215,537 (-44.0%) | 384,459 (-0.1%) | 196,479 (-48.9%) | 358,894 (-6.7%) |
| Federal student loan borrowers, New Hampshire | 196,264 | 104,938 (-46.5%) | 96,432 (-50.9%) | 196,352 (+0.0%) | 104,241 (-46.9%) | 190,750 (-2.8%) |
| Federal student loan borrowers, New Jersey | 1.29M | 450,356 (-65.1%) | 629,605 (-51.3%) | 1.31M (+1.3%) | 519,952 (-59.8%) | 1.22M (-5.8%) |
| Federal student loan borrowers, New Mexico | 236,312 | 153,772 (-34.9%) | 88,408 (-62.6%) | 236,428 (+0.0%) | 130,651 (-44.7%) | 233,314 (-1.3%) |
| Federal student loan borrowers, New York | 2.57M | 1.22M (-52.5%) | 1.19M (-53.7%) | 2.57M (+0.0%) | 1.25M (-51.3%) | 2.57M (+0.1%) |
| Federal student loan borrowers, North Carolina | 1.44M | 732,861 (-49.2%) | 950,046 (-34.1%) | 1.44M (+0.1%) | 789,713 (-45.2%) | 1.44M (-0.1%) |
| Federal student loan borrowers, North Dakota | 92,434 | 52,351 (-43.4%) | 64,873 (-29.8%) | 92,391 (-0.0%) | 56,986 (-38.3%) | 86,386 (-6.5%) |
| Federal student loan borrowers, Ohio | 1.85M | 699,900 (-62.2%) | 507,458 (-72.6%) | 1.75M (-5.3%) | 613,616 (-66.9%) | 1.57M (-15.2%) |
| Federal student loan borrowers, Oklahoma | 534,002 | 315,480 (-40.9%) | 256,421 (-52.0%) | 534,088 (+0.0%) | 278,947 (-47.8%) | 485,430 (-9.1%) |
| Federal student loan borrowers, Oregon | 550,628 | 312,082 (-43.3%) | 348,077 (-36.8%) | 551,112 (+0.1%) | 332,063 (-39.7%) | 550,033 (-0.1%) |
| Federal student loan borrowers, Pennsylvania | 1.94M | 633,413 (-67.3%) | 491,520 (-74.6%) | 1.96M (+1.2%) | 596,122 (-69.2%) | 1.75M (-9.6%) |
| Federal student loan borrowers, Rhode Island | 156,531 | 91,242 (-41.7%) | 70,034 (-55.3%) | 156,520 (-0.0%) | 85,691 (-45.3%) | 146,908 (-6.1%) |
| Federal student loan borrowers, South Carolina | 817,995 | 307,174 (-62.4%) | 272,651 (-66.7%) | 819,067 (+0.1%) | 268,406 (-67.2%) | 667,053 (-18.5%) |
| Federal student loan borrowers, South Dakota | 122,025 | 49,540 (-59.4%) | 35,772 (-70.7%) | 121,939 (-0.1%) | 44,993 (-63.1%) | 83,721 (-31.4%) |
| Federal student loan borrowers, Tennessee | 935,732 | 645,574 (-31.0%) | 728,891 (-22.1%) | 935,831 (+0.0%) | 686,035 (-26.7%) | 935,468 (-0.0%) |
| Federal student loan borrowers, Texas | 4.13M | 1.80M (-56.4%) | 1.74M (-57.8%) | 4.13M (+0.0%) | 1.90M (-54.0%) | 4.12M (-0.1%) |
| Federal student loan borrowers, Utah | 327,595 | 239,327 (-26.9%) | 276,270 (-15.7%) | 327,597 (+0.0%) | 277,588 (-15.3%) | 328,133 (+0.2%) |
| Federal student loan borrowers, Vermont | 78,213 | 50,912 (-34.9%) | 53,367 (-31.8%) | 78,213 (+0.0%) | 52,194 (-33.3%) | 78,167 (-0.1%) |
| Federal student loan borrowers, Virginia | 1.14M | 542,755 (-52.2%) | 569,480 (-49.8%) | 1.13M (-0.0%) | 542,429 (-52.2%) | 1.02M (-10.2%) |
| Federal student loan borrowers, Washington | 811,512 | 507,889 (-37.4%) | 394,762 (-51.4%) | 811,796 (+0.0%) | 465,433 (-42.6%) | 809,698 (-0.2%) |
| Federal student loan borrowers, West Virginia | 238,717 | 97,733 (-59.1%) | 110,576 (-53.7%) | 234,183 (-1.9%) | 104,567 (-56.2%) | 165,129 (-30.8%) |
| Federal student loan borrowers, Wisconsin | 741,455 | 322,918 (-56.4%) | 325,050 (-56.2%) | 741,856 (+0.1%) | 320,358 (-56.8%) | 740,079 (-0.2%) |
| Federal student loan borrowers, Wyoming | 57,300 | 33,514 (-41.5%) | 25,898 (-54.8%) | 57,400 (+0.2%) | 30,505 (-46.8%) | 56,980 (-0.6%) |
| Federal student loan balance, borrower age 24 and Younger | $84.9B | $52.1B (-38.6%) | $49.2B (-42.0%) | $84.9B (-0.0%) | $51.9B (-38.9%) | $84.8B (-0.1%) |
| Federal student loan balance, borrower age 25 to 34 | $462.0B | $216.2B (-53.2%) | $206.3B (-55.3%) | $461.9B (-0.0%) | $212.5B (-54.0%) | $460.8B (-0.2%) |
| Federal student loan balance, borrower age 35 to 49 | $679.5B | $353.4B (-48.0%) | $349.6B (-48.5%) | $677.9B (-0.2%) | $340.7B (-49.9%) | $672.9B (-1.0%) |
| Federal student loan balance, borrower age 50 to 61 | $310.2B | $168.5B (-45.7%) | $201.0B (-35.2%) | $310.2B (-0.0%) | $180.3B (-41.9%) | $310.1B (-0.0%) |
| Federal student loan balance, borrower age 62 and Older | $142.5B | $68.1B (-52.2%) | $80.0B (-43.9%) | $142.5B (-0.0%) | $70.3B (-50.6%) | $142.5B (-0.0%) |
| Federal student loan borrowers, borrower age 24 and Younger | 6.13M | 2.58M (-57.9%) | 2.57M (-58.2%) | 6.13M (+0.0%) | 2.65M (-56.8%) | 5.52M (-10.0%) |
| Federal student loan borrowers, borrower age 25 to 34 | 13.85M | 6.59M (-52.4%) | 6.72M (-51.5%) | 13.84M (-0.1%) | 6.66M (-51.9%) | 13.83M (-0.1%) |
| Federal student loan borrowers, borrower age 35 to 49 | 14.84M | 7.87M (-46.9%) | 8.08M (-45.5%) | 14.83M (-0.0%) | 8.00M (-46.1%) | 14.84M (+0.0%) |
| Federal student loan borrowers, borrower age 50 to 61 | 6.33M | 2.91M (-54.1%) | 2.98M (-52.9%) | 6.33M (+0.0%) | 2.88M (-54.6%) | 5.73M (-9.4%) |
| Federal student loan borrowers, borrower age 62 and Older | 3.16M | 1.23M (-61.2%) | 1.24M (-60.7%) | 3.13M (-1.1%) | 1.23M (-61.0%) | 2.75M (-13.2%) |
| Federal student loan balance, borrower debt size Less than 5K | $18.7B | $7.2B (-61.5%) | $7.3B (-61.2%) | $18.5B (-1.3%) | $7.4B (-60.2%) | $15.8B (-15.3%) |
| Federal student loan balance, borrower debt size 5K to 10K | $52.0B | $20.4B (-60.7%) | $18.5B (-64.4%) | $50.8B (-2.2%) | $19.5B (-62.6%) | $43.9B (-15.5%) |
| Federal student loan balance, borrower debt size 10K to 20K | $131.2B | $55.2B (-57.9%) | $56.1B (-57.2%) | $131.1B (-0.0%) | $57.0B (-56.5%) | $131.1B (-0.0%) |
| Federal student loan balance, borrower debt size 20K to 40K | $268.0B | $139.9B (-47.8%) | $155.4B (-42.0%) | $268.0B (-0.0%) | $145.5B (-45.7%) | $267.8B (-0.1%) |
| Federal student loan balance, borrower debt size 40K to 60K | $204.3B | $110.4B (-46.0%) | $108.8B (-46.7%) | $204.2B (-0.0%) | $110.9B (-45.7%) | $204.3B (+0.0%) |
| Federal student loan balance, borrower debt size 60K to 80K | $175.6B | $131.5B (-25.1%) | $144.6B (-17.7%) | $175.7B (+0.0%) | $137.1B (-21.9%) | $178.3B (+1.5%) |
| Federal student loan balance, borrower debt size 80K to 100K | $126.4B | $62.9B (-50.3%) | $55.8B (-55.8%) | $126.4B (-0.0%) | $59.6B (-52.9%) | $126.4B (+0.0%) |
| Federal student loan balance, borrower debt size 100K to 200K | $350.7B | $140.7B (-59.9%) | $145.0B (-58.7%) | $350.6B (-0.0%) | $136.2B (-61.2%) | $351.3B (+0.2%) |
| Federal student loan balance, borrower debt size 200K+ | $352.2B | $190.3B (-46.0%) | $194.6B (-44.8%) | $352.0B (-0.0%) | $182.6B (-48.2%) | $352.2B (+0.0%) |
| Federal student loan borrowers, borrower debt size Less than 5K | 6.82M | 2.70M (-60.4%) | 2.81M (-58.9%) | 6.65M (-2.6%) | 2.87M (-58.0%) | 5.96M (-12.7%) |
| Federal student loan borrowers, borrower debt size 5K to 10K | 7.12M | 2.89M (-59.4%) | 2.64M (-62.9%) | 7.12M (-0.0%) | 2.76M (-61.3%) | 6.16M (-13.5%) |
| Federal student loan borrowers, borrower debt size 10K to 20K | 9.10M | 3.82M (-58.0%) | 3.86M (-57.5%) | 8.97M (-1.4%) | 3.92M (-57.0%) | 8.98M (-1.4%) |
| Federal student loan borrowers, borrower debt size 20K to 40K | 9.40M | 5.06M (-46.1%) | 5.56M (-40.8%) | 9.39M (-0.1%) | 5.24M (-44.2%) | 9.39M (-0.0%) |
| Federal student loan borrowers, borrower debt size 40K to 60K | 4.15M | 2.27M (-45.3%) | 2.24M (-46.2%) | 4.19M (+0.9%) | 2.28M (-45.1%) | 4.20M (+1.1%) |
| Federal student loan borrowers, borrower debt size 60K to 80K | 2.57M | 1.91M (-25.8%) | 2.09M (-18.5%) | 2.56M (-0.6%) | 1.98M (-23.0%) | 2.57M (-0.0%) |
| Federal student loan borrowers, borrower debt size 80K to 100K | 1.38M | 728,545 (-47.4%) | 640,771 (-53.7%) | 1.45M (+4.5%) | 688,165 (-50.3%) | 1.45M (+4.9%) |
| Federal student loan borrowers, borrower debt size 100K to 200K | 2.57M | 1.09M (-57.6%) | 1.09M (-57.8%) | 2.58M (+0.3%) | 1.05M (-59.3%) | 2.64M (+2.7%) |
| Federal student loan borrowers, borrower debt size 200K+ | 1.19M | 699,994 (-41.0%) | 656,475 (-44.7%) | 1.36M (+14.3%) | 644,088 (-45.7%) | 1.32M (+11.3%) |
| Parent PLUS loan balance | $114.8B | $91.8B (-20.1%) | $102.5B (-10.8%) | $114.8B (+0.0%) | $93.2B (-18.9%) | $114.8B (-0.0%) |
| Federal student loan balance excluding Parent PLUS | $1,564.2B | $766.6B (-51.0%) | $783.6B (-49.9%) | $1,562.5B (-0.1%) | $762.6B (-51.2%) | $1,556.4B (-0.5%) |
