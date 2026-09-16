# Current ASEC interest observations

`current_asec_interest_source.qualify_current_asec_interest(preparation)`
qualifies ordinary and retirement-account interest from the original 2025 ASEC
person member, referring to income in 2024. It preserves the already qualified
`INT_VAL` combined total and exposes the published components and their
reporting, account, allocation and disclosure metadata.

The source is the [2025 ASEC public-use data dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf),
SHA-256 `5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f`.
The exact total-income domain is reused from the pinned
`asec_current_money_domains_v1.json`; the additional fields use this same
dictionary. Pages below are one-based PDF pages.

| Fields | Meaning and printed universe | Page |
| --- | --- | --- |
| `INT_VAL`, `INT_YN` | Combined interest; amount universe `INT_YN = 1`, receipt universe age 15+ | 46 |
| `TRDINT_VAL` | Interest excluding retirement-account interest; `INT_YN = 1` | 50 |
| `RINT_YN` | Retirement-account interest receipt; age 15+ | 49 |
| `RINT_SC1`, `RINT_SC2` | Account identities, including 401k, 403b, Roth IRA, Regular IRA, KEOGH, SEP and other; `RINT_YN = 1` | 49 |
| `RINT_VAL1`, `RINT_VAL2` | Interest in the two reported account slots; corresponding account code greater than zero | 49 |
| `I_INTVAL`, `I_INTYN` | Combined-interest allocation flags, with distinct composite code systems | 56 |
| `I_RINTSC`, `I_RINTVAL1`, `I_RINTVAL2`, `I_RINTYN` | Retirement-account allocation flags; `I_RINTSC` names slot 1 only | 57 |
| `TRINT_VAL1`, `TRINT_VAL2`, `TTRDINT_VAL` | Published top-code indicators for the respective positive component amounts | 60 |

The account, receipt and applicable allocation code systems reuse the income
routing qualifier. `I_INTVAL` accepts the printed codes 0 and 11–15; its header
interval does not make codes 1–10 valid. There is no individually published
allocation flag for `TRDINT_VAL` or `RINT_SC2` in this dictionary. Allocation
labels describe publisher processing, not observation validity or tax status.

## Amounts and knownness

The result keeps raw source literals, parsed published amounts, parse status,
reporting status and canonical amount knownness separately. Missing and
malformed components remain unknown. Interest amounts must be nonnegative
integers within each field's published encoding; these disclosure ranges are
not latent-income bounds.

`TRDINT_VAL` describes its range as a dollar value, so a published zero with
combined-interest receipt `yes` is an observed zero ordinary component.
`INT_VAL` and `RINT_VAL1/2` describe zero as none or NIU, so zero with receipt
`yes` remains ambiguous. A retirement-interest recipient with account code 0
and slot amount 0 has an `unreported_account_slot`, not an imputed account
amount. Receipt `no` plus account code 0 and amount 0 establishes known
nonreceipt. Under-15 NIU values never become analytical zeros.

The diagnostic `combined_minus_published_components` subtracts the three
parsed published component cells from `INT_VAL`. Its knownness means that all
four published numbers are readable; it does not establish that all three
components are reported observations. `all_component_amounts_observed`
separately records that stronger condition. The diagnostic may be negative;
the qualifier never balances, clips or replaces any published number.

## Source lifetime and composition

Qualification borrows the actual `AuthenticatedSurveyPopulationPreparation`
and its retained native ASEC/current-money owners. It matches the original
member's archive digest, member digest, byte count and row count, captures that
member once, and verifies the captured file after reading. Current-cohort
`PERIDNUM` keys must form an exact roster; household coordinates, person line
numbers and raw ages must match the retained parent. Source row order is
irrelevant. The original `INT_VAL` validity and float64 bits must agree with
the current-money owner before selected ASEC people are projected.

`CurrentAsecInterestValues(person, asec_literals, evidence)` is descriptive
transport. Its seal covers both complete tables, axes, nullable masks and
backing storage, exact float bits and evidence. The qualifier rechecks the
actual preparation after its last source I/O and checks the complete value
seal before returning. A consumer must retain the actual preparation,
requalify it immediately before consumption and after its own last relevant
I/O, and compare the full value seal before returning or exporting a
successor. A copied dataclass, evidence dictionary or matching digest cannot
grant source authority.

The only existing helper change is an optional explicit column/amount-grammar
roster on the private income-routing literal reader. Its original defaults
remain in effect for the income-routing qualifier. This module adds no host,
run issuer, graph execution or model fitting.

## Bridge work still required

[ACS response guidance](https://www.census.gov/programs-surveys/acs/respond/get-help.html)
includes interest credited to retirement accounts in its broad
interest/dividend/property item. The live guide currently refers to the 2025
questionnaire; it is not an archived 2024 instruction artifact. This
ASEC qualification therefore keeps ordinary interest and retirement-account
interest separate for a later, reviewed measurement bridge. It does not
declare either component taxable, tax-exempt, regularly withdrawn, or equal
to an ACS component. It does not alter tax leaves, original or PUF clones,
financial imputation, calibration or release eligibility.
