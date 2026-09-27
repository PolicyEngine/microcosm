# PUF55 survey Social Security conditioning decision

On 2026-09-09 the build owner adopted
`social_security_filer_joint_spouse_report_sum_proxy` for PUF matching. This
supersedes the pending measurement judgment in
[survey-social-security-source.md](survey-social-security-source.md). The
decision authorizes this conditioning measurement and its explicit fallback;
it does not assert donor/recipient equivalence, allocate beneficiaries, complete
Social Security components, approve a fit, or admit a release.

For each modeled return, include exactly one `HEAD` and, only when the actual
`filing_status_input` is `JOINT`, exactly one `SPOUSE`. Sum their available
nonnegative source report totals. Preserve combined family and child reports
without splitting, deduplicating, or subtracting amounts. Exclude dependents
from this return feature without changing any person's observed or unknown
Social Security cells or component masks.

The public helper requires the actual authenticated survey preparation and
qualifies current source reports. It uses the exact modeled return roles
retained by that preparation. It does **not** rerun tax-unit construction using
current money. That limitation is accepted for this conditioning stage; later
production integration must qualify the chosen role/current-money relationship
explicitly. A caller-made role table or decoded receipt grants no authority.
The PUF disclosure-capped return-size predictor is not a person count, and the
coarsened PUF filing class 2 cannot identify an actual spouse: surviving-spouse
returns also map to that class.

## Availability and refusal

An available report-sum means every included reporter has an in-universe,
finite source amount. It does not mean those reporters legally own all the
reported benefits. Known zero reports are valid. Unknown components do not
invalidate a known total. Family/child ambiguity remains an explicit limitation
and does not itself trigger missingness.

An unavailable included report, including an under-15 filer outside the source
question universe, leaves the proxy unknown and selects the separately named
`PUF55_SURVEY_SS_NO_TOTAL` profile. This profile has the same 55 outputs and
eight predictors. The existing `PUF55_SURVEY_SS` retains nine predictors.
Neither produces the four Social Security components. There is no partial sum,
dependent-zero convention, component-sum fallback, silent row deletion, or
implicit change to the 59/65-output profiles.

Missing/invalid roles, duplicate or orphan membership, conflicting source
values, nonnumeric amounts, negative/nonfinite observations, and known amounts
outside the reporting universe refuse. They cannot route to eight predictors:
the first two PUF predictors still require valid roles. A source qualifier's own
refusals remain refusals; the fallback does not override them. Graph integration
must bind both disjoint recipient routes, account for each recipient exactly
once, authenticate inherited clone membership, and execute the corresponding
complete fit/apply/attachment chain. This numerical helper does none of that
graph admission or fitting.

## Measurement limits

[IRS Publication 915 (2015), pages 2–6](https://www.irs.gov/pub/irs-prior/p915--2015.pdf#page=2)
assigns benefits to the legal beneficiary, including a child whose parent
receives the check; joint filers combine spouses' benefits. The return amount
uses benefits net of repayments and includes the Social Security equivalent
part of tier-1 Railroad Retirement, excluding other pension portions. Form
1040EZ omits benefits, and some recipients need not file. These rules motivate
the selected return members but do not establish survey report ownership or
PUF editing conventions. [SSA's representative-payee guidance](https://www.ssa.gov/payee/faqrep.htm)
likewise distinguishes the person managing a payment from its beneficiary.

The [2025 ASEC dictionary, PDF pages 48–49](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf#page=48)
permits combined family payments and identifies payments on behalf of children.
Its source total covers calendar 2024 for a 2025 interview. The
[ACS 2024 questionnaire, question 43](https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2024/quest24.pdf#page=18)
uses the prior twelve months, permits jointly received income entirely on one
person's record, and combines Social Security with Railroad Retirement.
[ACS SSP metadata, PDF page 45](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf#page=45)
defines the age-15 universe, rounded/topcoded amounts and ADJINC price
adjustment. The helper preserves those distinct observation windows; it does
not calendarize ACS or estimate repayments/Railroad adjustments.

The ninth donor predictor remains the existing transported 2015 `E02400`
carrier under its declared 2024 money-growth convention. Carrier zeros do not
prove absent benefits. Exact 2015 publisher editing/omission rules remain
unverified; the [IRS PUF page](https://www.irs.gov/statistics/soi-tax-stats-individual-public-use-microdata-files)
reports the 2012–2015 files unavailable. Matching is therefore conditional on
this documented proxy assumption, not a newly authenticated observed 2024
return total. Using eight predictors does not remove donor coverage concerns
for nonfilers or resolve assumptions in the other donor outcomes.

## Diagnostics required before relying on a fit

Family payments on selected reporters may overstate their own benefits;
payments reported elsewhere may understate them. An all-member sum includes
dependent reports and cannot solve this attribution problem. Topcoding,
allocation, return editing and transported donor money can distort zeros and
tails. Growth of fixed 2015 amounts does not predict 2024 cohort composition.

Report route counts, known zero/positive distributions and tails by survey,
actual filing status, age, child-report reasons, allocation provenance, and
multiple-return household status. Preserve original-household holdouts across
clones. Compare design-weighted held-out performance and recipient overlap for
eight versus nine predictors; donor-only validation cannot establish cross-source
measurement validity. An adult in-universe all-member report sum may be a
labelled sensitivity diagnostic, with under-15 unknownness retained. It is not
the default feature. Benefit-component modeling and release checks remain
separate work.

This change provides source-bound numerical values, explicit route metadata,
and an isolated graph profile declaration. Actual source guards, route-aware
graph integration, donor projection for the fallback, fits and diagnostics are
subsequent verification steps, not results claimed by this decision.
