# Processed PUF 1.8.0 artifact audit

This audit covers the exact
`release://policyengine/irs-soi-puf/1.8.0/puf_2024.h5` asset used by the US
build. It separates the bytes in that release from the logical dataset that a
later, retired `policyengine-us-data` class produced by mutating those bytes on
load.

No microdata values are recorded here. Row counts, file hashes, and other
aggregate metadata are safe audit evidence.

## Pinned bytes

| Property | Audited value |
|---|---:|
| File size | 241,045,964 bytes |
| SHA-256 | `8182579ddfecaf5e5b872e2307b88f03e8e8def993171b648f701a19a847f37b` |
| MD5 | `d18cebd81844e67350d58324156fd196` |
| Root datasets | 74 |
| Root attributes | 0 |
| Person rows | 484,015 |
| Tax-unit / household rows | 207,692 |
| Marital-unit rows | 362,844 |

There are 52 person-length arrays, 21 tax-unit-length arrays, and one
marital-unit-length array.

The separately pinned `puf_2015.h5` has the same 74 names, shapes, dtypes, and
241,045,964-byte size, but a distinct SHA-256
(`5144bf2eafa705216e157858ec05c84f930cb9e2785abfc676b81216e8b68bc3`)
and MD5 (`13a07cba76cdb982c0f0a13ca0497b4b`). Neither file is raw IRS
microdata: both are PolicyEngine entity-array exports.

The physical release matches the generator at the local `1.8.0` tag,
commit `371f77a0aadfdeacd5856e0a3030c2db0eda65b5`. That generator:

- reads the raw `IRS_PUF_2015` tables, renames and derives PolicyEngine inputs,
  and expands tax units into entities (`datasets/puf/puf.py` at the tag,
  lines 134-223 and 287-393);
- randomly allocates person-grain financial inputs between filer and spouse
  using `EARNSPLIT` (`puf.py` at the tag, lines 395-462);
- creates 2024 by loading the processed 2021 arrays and applying
  PolicyEngine-variable uprating factors (`puf.py` at the tag, lines 299-315).

The current Populace donor immediately regroups person arrays on
`person_tax_unit_id`. The retired filer/spouse allocation therefore cancels
for the consumed financial totals; it remains retired content in the file,
but it does not change the tax-unit signal Populace fits.

## Classification rule

- **A — raw field passthrough, possibly uprated**: the consumed tax-unit total
  is one raw IRS PUF field (including `P*`, `T*`, `S006`, and `MARS`-adjacent
  identity fields where stated), even if the release renamed, uprated, or
  temporarily split it over people.
- **B — retired-code-derived**: the consumed value combines fields, maps an
  IRS code, constructs an identity, or applies a retired simulation/proxy.
- **C — unused**: bare h5py loads the array into memory, but
  `puf_tax_unit_donor_from_arrays` does not consult it under the production
  default output contract.

This is a physical-artifact classification. Populace logic applied after the
source boundary, such as the E19200 mortgage carve or QRF imputation, is not
attributed to the artifact.

## A — consumed raw-field passthroughs

All 34 columns in this table are present in the physical HDF and logically
consumed. `P` means 484,015 person rows and `TU` means 207,692 tax-unit rows.
The release mappings are at `datasets/puf/puf.py` at tag `1.8.0`, lines
134-200; the corresponding later archived mappings are at commit
`42ed5d45c56df80d754fbe24cce21cfeb8d05cbe`, lines 636-719.

| HDF column | Grain | Raw IRS PUF source | Current Populace use |
|---|---:|---|---|
| `alimony_expense` | P | `E03500` | `alimony_expense` |
| `alimony_income` | P | `E00800` | `alimony_income` |
| `casualty_loss` | P | `E20500` | `casualty_loss` |
| `charitable_cash_donations` | P | `E19800` | same-name output |
| `charitable_non_cash_donations` | P | `E20100` | same-name output |
| `domestic_production_ald` | TU | `E03240` | same-name output |
| `educator_expense` | P | `E03220` | same-name output |
| `employment_income` | P | `E00200` | `employment_income_before_lsr` and predictor |
| `farm_income` | P | `T27800` | same-name Schedule J output |
| `farm_rent_income` | P | `E27200` | same-name output |
| `health_savings_account_ald` | TU | `E03290` | same-name output |
| `household_weight` | TU | `S006 / 100`, then return-count uprating | donor design weight |
| `investment_income_elected_form_4952` | P | `E58990` | same-name output |
| `long_term_capital_gains` | P | `P23250` | `long_term_capital_gains_before_response` |
| `long_term_capital_gains_on_collectibles` | P | `E24518` | same-name output |
| `miscellaneous_income` | P | `E01200` | same-name output |
| `non_sch_d_capital_gains` | P | `E01100` | same-name output |
| `partnership_s_corp_income` | P | `E26270` in release 1.8.0 | partnership fallback; S-corp fallback is zero |
| `qualified_dividend_income` | P | `E00650` | same-name output and predictor |
| `qualified_tuition_expenses` | P | `E03230` in release 1.8.0 | same-name output |
| `real_estate_taxes` | P | `E18500` | same-name output |
| `salt_refund_income` | P | `E00700` | same-name output |
| `self_employed_pension_contribution_ald` | TU | `E03300` | `self_employed_pension_contributions_desired` |
| `self_employment_income` | P | `E00900` | `self_employment_income_before_lsr` and predictor |
| `short_term_capital_gains` | P | `P22250` | same-name output and predictor |
| `social_security` | P | `E02400` | retirement component; other components are zero |
| `student_loan_interest` | P | `E03210` | same-name output |
| `tax_exempt_interest_income` | P | `E00400` | same-name output |
| `tax_unit_id` | TU | `RECID` | donor tax-unit identity |
| `taxable_interest_income` | P | `E00300` | same-name output and predictor |
| `taxable_ira_distributions` | P | `E01400` | same-name output |
| `taxable_pension_income` | P | `E01700` | `taxable_private_pension_income` |
| `traditional_ira_contributions` | P | `E03150` | `traditional_ira_contributions_desired` |
| `unrecaptured_section_1250_gain` | TU | `E24515` | same-name output |

The named direct-mapping cases are therefore unambiguous:

- educator expense is an **A** passthrough from `E03220`;
- Form 4952 elected investment income is an **A** passthrough from `E58990`;
- SALT-refund income is an **A** passthrough from `E00700`;
- the physical 1.8.0 qualified-tuition array is an **A** passthrough from
  `E03230`, not the later fallback.

Commit `42ed5d45...` later changed qualified tuition to
`max(E03230, E87530)` (`datasets/puf/puf.py` lines 584-589 and 668). That
fallback is retired-code-derived, but it is not embedded in this older release
asset. Populace has already made it explicit source-stage logic for a future
raw pin; when given only the processed array, the current donor consumes that
array and cannot exercise the fallback.

## B — consumed retired-code-derived columns

| HDF column | Grain | Retired derivation | Archived evidence |
|---|---:|---|---|
| `estate_income` | P | `E26390 - E26400` | tag-1.8.0 `puf.py` line 148; `42ed5d45...` lines 650-651 |
| `filing_status` | TU | map `MARS` 1/2/3/4 to four status strings | tag-1.8.0 `puf.py` lines 211-218; `42ed5d45...` lines 789-796 |
| `non_qualified_dividend_income` | P | `E00600 - E00650` | tag-1.8.0 `puf.py` line 162; `42ed5d45...` line 666 |
| `person_tax_unit_id` | P | repeat the `RECID`-based tax-unit identity over constructed people | tag-1.8.0 `puf.py` lines 419-479; `42ed5d45...` lines 1530-1628 |
| `rental_income` | P | `E25850 - E25860` | tag-1.8.0 `puf.py` line 167; `42ed5d45...` lines 673-674 |
| `w2_wages_from_qualified_business` | P | `0.16 * max(0, E00900 + E26270 + E02100 + E27200)` | tag-1.8.0 `puf.py` lines 203-206 |

The physical W-2 proxy is especially important: it is not the later
YAML-driven Section 199A v1 simulation.

As an aggregate cross-check, regrouping the five named person arrays to tax
units and weighting them with the artifact's 161,180,574.72 total design
weight gives:

| Column | Weighted nonzero share | Weighted mean | Mean when positive |
|---|---:|---:|---:|
| `educator_expense` | 2.4826% | $8.56 | $344.72 |
| `investment_income_elected_form_4952` | 0.1394% | $32.25 | $23,136.64 |
| `salt_refund_income` | 13.5796% | $274.41 | $2,020.76 |
| `qualified_tuition_expenses` | 1.0961% | $35.40 | $3,229.67 |
| `w2_wages_from_qualified_business` | 15.9032% | $2,298.33 | $14,451.96 |

## C — physically present but unused columns

These 34 columns complete the enumeration. The reason in every row is
"outside the production default donor contract" unless a more specific
contract mismatch is called out.

| HDF column | Grain | Audit note |
|---|---:|---|
| `age` | P | unused demographic |
| `american_opportunity_credit` | TU | formula/output, not a persisted donor leaf |
| `amt_foreign_tax_credit` | P | outside contract |
| `cdcc_relevant_expenses` | TU | outside contract |
| `early_withdrawal_penalty` | P | outside contract |
| `energy_efficient_home_improvement_credit` | TU | formula/output, not a persisted donor leaf |
| `excess_withheld_payroll_tax` | P | outside contract |
| `family_id` | TU | unused entity identity |
| `foreign_tax_credit` | TU | formula/output, not a persisted donor leaf |
| `general_business_credit` | P | outside contract |
| `household_id` | TU | unused entity identity |
| `interest_deduction` | TU | current contract requires mortgage-only structural inputs instead |
| `is_male` | P | unused demographic |
| `is_tax_unit_dependent` | P | unused relationship flag |
| `is_tax_unit_head` | P | unused relationship flag |
| `is_tax_unit_spouse` | P | unused relationship flag |
| `marital_unit_id` | MU | unused entity identity |
| `misc_deduction` | TU | current contract requires `unreimbursed_business_employee_expenses` |
| `other_credits` | P | outside contract |
| `person_family_id` | P | unused entity link |
| `person_household_id` | P | unused entity link |
| `person_id` | P | unused entity identity |
| `person_marital_unit_id` | P | unused entity link |
| `person_spm_unit_id` | P | unused entity link |
| `pre_tax_contributions` | P | separate retirement-input construction owns the current leaves |
| `prior_year_minimum_tax_credit` | P | outside contract |
| `recapture_of_investment_credit` | TU | outside contract |
| `savers_credit` | TU | formula/output, not a persisted donor leaf |
| `self_employed_health_insurance_ald` | TU | formula-owned aggregate; explicitly blocked |
| `spm_unit_id` | TU | unused entity identity |
| `state_and_local_sales_or_income_tax` | TU | outside current persisted-output contract |
| `tax_exempt_pension_income` | P | outside contract |
| `taxable_unemployment_compensation` | P | outside contract |
| `unreported_payroll_tax` | TU | outside contract |

`P` and `TU` in this table have the same row counts as above; `MU` has 362,844
rows.

## The 15-leaf QBI surface is a load-time mutation

The exact HDF contains only one of the current 15 QBI-contract names:
`w2_wages_from_qualified_business`. It has no
`qbi_simulation_version` attribute.

At archived commit `42ed5d45...`, `datasets/puf/puf.py` lines 105-405 define a
new seeded v1 model, lines 748-787 apply it during fresh preprocessing, and
lines 860-879 list the exported leaves. But release 1.8.0 predates that model.
To bridge the mismatch, the retired `PUF` class:

1. detects missing leaves or a missing version attribute;
2. reconstructs source arrays and runs the v1 draws;
3. opens the downloaded release file in `r+`;
4. writes the missing/replaced arrays and stamps
   `qbi_simulation_version = 1`.

That migration is `puf.py` lines 993-1302 and is invoked from `load` /
`load_dataset` at lines 1355-1368. Thus the often-observed 15-leaf "1.8.0
PUF" was a post-download, locally mutated logical view, not the pinned
artifact.

Populace reads the file with bare h5py
(`tools/build_us_puf_support_base.py`, `_read_h5_arrays`) and does not run the
retired migration. Against the exact bytes, the current default donor contract
is missing 24 outputs:

- 14 of the 15 QBI leaves;
- `home_mortgage_interest`,
  `unreimbursed_business_employee_expenses`, `farm_operations_income`, and
  `partnership_self_employment_net_earnings`;
- six structural mortgage balance/interest/origination-year leaves.

This audit therefore treats the new Populace QBI simulation stage as a port of
retired load-time logic, not as a redraw of values already present in the
pinned artifact.

## Ready-to-paste PR-body summary

The pinned `puf_2024.h5` audit found 74 arrays: 40 are logically consumed by
the current PUF donor path (34 raw-field passthroughs after tax-unit regrouping
and six retired-generator derivations), while 34 are unused. Educator
expense (`E03220`), Form 4952 elected investment income (`E58990`), and SALT
refunds (`E00700`) are direct raw-field mappings. The physical 1.8.0 tuition
array is also direct `E03230`; the `max(E03230, E87530)` fallback was added
later and now lives explicitly in Populace's raw-source transformation.

The release bytes do **not** contain the claimed 15-leaf QBI v1 surface. They
contain only an older deterministic W-2 proxy and no QBI version attribute.
Archived `policyengine-us-data` later made the surface appear complete by
opening the release file read-write, simulating 14 missing leaves (and
replacing stale leaves), and stamping version 1 during `PUF.load()`. Populace's
bare-HDF path never invokes that side effect. This PR ports that versioned
simulation into repository-owned logic and separately documents the move from
processed entity arrays to raw IRS PUF inputs plus Populace-owned aging.
