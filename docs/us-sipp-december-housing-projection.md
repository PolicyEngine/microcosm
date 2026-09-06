# December SIPP housing source projection

`microcosm.build.sipp_housing_measurements` measures December housing evidence
from explicitly supplied 2023 SIPP rows,
`microcosm.build.sipp_housing_predictors` observes the version 3 predictor
columns on the roster that measurement already fixed, and
`microcosm.build.sipp_housing_projection` declares the versioned recipe that
says which lexical columns such an extraction must carry and what each column's
published contract is.

Neither module opens a file, authenticates bytes, publishes an artifact, fits a
model, maps a recipient variable, annualizes an amount, rebases a price or
transfers a population. **No genuine extraction has been run.** The runnable
first-32-SSUID proposal is described at the end of this document and is executed
only after review.

## The frozen sixteen-field producer is untouched

`microcosm.build.sipp_december_source` and `microcosm.build.sipp_measurements`
keep every byte. Their module hashes are part of the producer binding that
`_contract()` publishes, and that binding is inside the projection and receipt
keys of receipts that already exist, so adding a housing field to the closed
`_HOUSEHOLD` registry would invalidate them. `FIELDS`, `_COPIED`, `_PINS`, the
`(39513, 17646, 17314, 318, 29)` reconciliation counts and the
`sipp_december_source_v1` receipt format are unchanged, and
`test_us_sipp_housing_projection.py` asserts each of them.

The housing receipt binds all five modules' bytes (see "Implementation binding"
below). Binding is a read: it records what the frozen modules are, and moves the
*housing* receipt if they ever change. It does not alter them, and the frozen
producer's own `_contract()` binding is untouched.

The housing modules therefore import those internals rather than copying them.
The kernel reuses `_number`, `_flag`, `_raw`, `_prepare`, `_check_source`,
`_QUALITY`, the fixed arithmetic context and `december_households`; the recipe
reuses `_PINS`, `_SOURCE_NAME`, `_MODES`, `_key`, `_sha`, `_decimal`,
`_measurement`, `_membership` and `_weights`. `december_housing()` regroups the
same supplied rows and refuses `HOUSING_ROSTER_DISAGREEMENT` if its grouping
differs from `december_households`, so the housing lane cannot quietly adopt a
different household identity.

## Provenance

Every contract below was read from the official
[2023 SIPP data dictionary](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf),
SHA-256 `0f1c5fbd29b24d3f1c18e7168a0f6d9fdfa7578b58d247fa9f475093a6db894f`,
3,991,831 bytes, 3,371 pages — the same digest the frozen producer already pins
as `dictionary_sha256`. `verify_dictionary()` refuses if the two ever diverge,
so this lane introduces no second document authority.

Page locators use **zero-based PDF page P**; the printed page is P+1 and an
ordinary viewer shows P+1. For every field in both tables below the printed page
number was read off the foot of that page in the PDF itself, not inferred from
where a name falls in a text rendering: a text-boundary guess and the page a
field's `Name:` line actually sits on are different claims, and only the second
is transcribed here. Column presence was confirmed against the pinned parent's
pipe-delimited header only. No SIPP data row was read to write this.

`EEHC_RENTSUB`'s universe is transcribed with straight quotes as
`TEHC_ST <= '60'`; the dictionary's own rendering uses typographic quotes. That
is quote normalization in the transcription, not a difference in the condition.

## The transcribed field table

| Field | P / printed | Status flag | File | Domain |
| --- | --- | --- | --- | --- |
| `TRENTMORT` | 379 / 380 | `ARENTMORT` | Household | $0–$9,999,999, length 7 |
| `TUTILS` | 380 / 381 | `AUTILS` | Household | $0–$9,999,999, length 7 |
| `ETENURE` | 784 / 785 | `ATENURE` | Household | 1, 2, 3 |
| `ERENTSUB` | 788 / 789 | `ARENTSUB` | Household | 1, 2 |
| `EVOUCHER` | 789 / 790 | `AVOUCHER` | Household | 1, 2 |
| `EEHC_TEN` | 786 / 787 | `AEHC_TEN` | Person-month | 1, 2, 3 |
| `EEHC_RENTSUB` | 790 / 791 | `AEHC_RENTSUB` | Person-month | 1, 2 |
| `EEHC_VOUCHER` | 791 / 792 | `AEHC_VOUCHER` | Person-month | 1, 2 |
| `EPRDEBT` | 429 / 430 | `APRDEBT` | Household | 1, 2 |
| `EMHDEBT` | 367 / 368 | `AMHDEBT` | Household | 1, 2 |
| `TRNTMT_STD` | 387 / 388 | none (`NA`) | Person | $1–$9,999,999 |
| `TRNTMT_MED` | 388 / 389 | none (`NA`) | Person | $1–$9,999,999 |
| `TUTILS_MED` | 389 / 390 | none (`NA`) | Person | $0–$9,999,999 |
| `TUTILS_STD` | 390 / 391 | none (`NA`) | Person | $0–$9,999,999 |

Each flag name was read off the dictionary entry's own **Status Flag** line, not
derived from the `T*`/`A*` convention that the four vehicle fields happen to
follow. `ARENTMORT` and `AUTILS` are what those pages literally print.

### The version 3 predictor columns

| Field | P / printed | Status flag | File | Length | Domain |
| --- | --- | --- | --- | --- | --- |
| `TAGE_EHC` | 796 / 797 | none (`NA`) | Person-month | 2 | 0–90 |
| `RHNUMPER` | 890 / 891 | `AHNUMPER` | Person-month | 2 | 1–20 |
| `RHNUMU18` | 891 / 892 | `AHNUMU18` | Person-month | 2 | 0–20 |
| `THTOTINC` | 2865 / 2866 | `AHTOTINC` | Person-month | 12 | −$10,000,000,000–$10,000,000,000 |
| `TEHC_ST` | 3310 / 3311 | `AEHC_ST` | Person-month | 2 | 53 published codes |

All five publish the universe `THHLDSTATUS in (1,2,3,4)`. `THHLDSTATUS` is not
carried by this projection, so that gate is transcribed and **not evaluated**,
exactly as it already is for `ETENURE`. Their descriptions, verbatim from the
pages above: "Monthly age during the reference period.", "Number of persons in
household this month", "Number of persons in household under 18 years this
month", "Sum of monthly earnings and income received by household members age 15
and older, as well as SSI payments received by children under age 15", "Monthly
state of residence".

`AEHC_ST`, `AHNUMPER`, `AHNUMU18` and `AHTOTINC` appear in the dictionary only
as **Status Flag** lines on their value fields' pages, with no entry of their
own. That is the same shape as `ATENURE`, `ARENTMORT`, `AUTILS`, `ARENTSUB` and
`AVOUCHER`, which the version 2 contract already carries, so it is the norm for
this document rather than a gap in these five.

Published universes, transcribed:

| Field | Universe |
| --- | --- |
| `TRENTMORT` | `ETENURE = 2 or (ETENURE = 1 and (EPRDEBT = 1 or EMHDEBT = 1))` |
| `TUTILS` | `ETENURE in (1,2,3)` |
| `ETENURE` | `THHLDSTATUS in (1,2,3,4)` |
| `ERENTSUB` | `ETENURE in (2,3)` |
| `EVOUCHER` | `ETENURE in (2,3) and ERENTSUB = 1` |
| `EEHC_TEN` | `TPRVLVQRT IN (1,2)` |
| `EEHC_RENTSUB` | `EEHC_TEN in (2,3) and TEHC_ST <= '60'` |
| `EEHC_VOUCHER` | `EEHC_RENTSUB = 1` |
| `EPRDEBT` | `TPRVAL > 0` |
| `EMHDEBT` | `TMHVAL > 0` |

Tenure codes are `1` "Owned or being bought by someone in the household?",
`2` "Rented?", `3` "Occupied without payment of rent?". The four indicator
fields are `1` "Yes", `2` "No". The topcode companions carry `Universe: NA` with
the universe description "Amount household paid for rent or mortgage was
topcoded" (or the utilities wording), so a populated value marks that person's
record as belonging to a topcoded group. **No topcode threshold is inferred.**

### No documented numeric missing sentinel

The dictionary's Not-in-Universe note, zero-based page 7, says that `.N`/`N`
describe **internal** SIPP data. The public pipe-delimited file's missing
encoding is not documented there, so the amount contracts declare
`documented_numeric_missing_sentinel: null`. `_number` requires a sentinel, so
one outside every transcribed domain is supplied and the resulting
`missing_item` class is relabelled `invalid_domain`: a token such as `-999` is
reported as out of domain, never as a documented missing code.

## The 38-column projection

```
SSUID PNUM MONTHCODE ERESIDENCEID WPFINWGT
ETENURE ATENURE TRENTMORT ARENTMORT TUTILS AUTILS
ERENTSUB ARENTSUB EVOUCHER AVOUCHER
EPRDEBT APRDEBT EMHDEBT AMHDEBT
EEHC_TEN AEHC_TEN EEHC_RENTSUB AEHC_RENTSUB EEHC_VOUCHER AEHC_VOUCHER
TRNTMT_MED TRNTMT_STD TUTILS_MED TUTILS_STD
TEHC_ST AEHC_ST RHNUMPER AHNUMPER RHNUMU18 AHNUMU18
THTOTINC AHTOTINC TAGE_EHC
```

The first five are the frozen producer's identity and weight columns, reused
unchanged. `ERESIDENCEID` remains the residence key, bounded 100001–400999.
**`SHHADID` is deliberately absent**: a second household identity would need its
own authority and reconciliation.

`recipe_document()["field_contracts"]` records why every column past the seven
mandated fields is carried. `EPRDEBT`/`EMHDEBT` resolve the owner branch of
`TRENTMORT`'s universe, so an owner household's exclusion reason is determinate
rather than "unresolved"; their own universes are `TPRVAL > 0` and `TMHVAL > 0`,
so an owner with neither reported is recorded as `owner_debt_unresolved` instead
of assumed debt-free. `EEHC_TEN` is the person-month tenure gate of
`EEHC_RENTSUB`'s universe. The topcode companions are what section 6.5 of the
requirements needs. `TEHC_ST`, `TPRVLVQRT` and `THHLDSTATUS` are **not** carried,
so the residence-state arm of `EEHC_RENTSUB`'s universe stays unseparated from
plain missingness at that grain — see "The state arm, now observed" below, which
changes no classification.

`TAGE_EHC` is the one column shared with the frozen sixteen-field producer,
which reads it as `_number(..., 0, 90, -999)` to gate its retirement-income and
bank-asset universes. The domain and the out-of-domain token are identical here,
so the shared column has one contract and not two. The projection is otherwise
disjoint from that producer's field list.

## Grain

`ETENURE`, `TRENTMORT`, `TUTILS`, `ERENTSUB`, `EVOUCHER`, `EPRDEBT` and
`EMHDEBT` are household-level values copied to every member; they go through the
frozen kernel's copy-agreement rule, and a within-household disagreement is
`household_copy_conflict` with an unknown measurement, exactly as the vehicle
fields behave. `EEHC_TEN`, `EEHC_RENTSUB` and `EEHC_VOUCHER` are person-month
values that are **never** copied: members may legitimately differ, which is
reported as a member disagreement rather than a conflict, and they never set the
household's assistance class. The topcode companions are person-level.

The five version 3 value columns are all printed **Person-month**.
`RHNUMPER`, `RHNUMU18` and `THTOTINC` describe household quantities on those
person-month records, so they go through the same copy-agreement rule and their
grain is recorded as `household_concept_person_month` — a file class of
Person-month with a household concept, stated as two facts rather than collapsed
into one. `TEHC_ST` goes through the copy-agreement rule too, but for a
different reason and with a different declaration
(`household_copy_agreement_checked_not_assumed`): a December residence has one
location, and the dictionary does **not** say the column is constant within a
household, so agreement is checked and a disagreement is a conflict, never a
majority vote. `TAGE_EHC` is per person and is never copied.

## Rent, mortgage and utilities

`TRENTMORT` is rent **or** mortgage. It is admitted as a contract rent paid only
when `ETENURE = 2`. On an owner household it is a mortgage payment and is
refused with `mortgage_excluded`; an owner is never turned into a zero-rent
renter. `ETENURE = 3` is outside `TRENTMORT`'s published universe and is refused
too. A renter's `TRENTMORT = 0` is a distinct, admitted case counted separately
from a missing one.

`TUTILS` is a separate tenant-paid utility amount with its own validity. It is
never added into a rent quantity, and it must not be added again anywhere a
gross-rent concept already includes utilities. In the PolicyEngine-US source
read for this lane — local checkout `686c21ec89`, version 1.779.4, inside this
workspace's `policyengine-us>=1.745.0,<2` range —
`policyengine_us/variables/gov/hud/hud_gross_rent.py` declares
`adds = ["pre_subsidy_rent", "hud_utility_allowance"]`, so a utility-inclusive
charge would double-count there. The engine is not installed in this worktree,
and that file was read rather than executed.

An assisted household's `TRENTMORT` is its post-assistance tenant payment. The
difference between it and a full contract charge is **not observed anywhere in
this source**: neither `ERENTSUB` nor `EVOUCHER` carries a subsidy amount, and no
landlord payment, program type or utility-reimbursement split exists in the
public file. Any statement that SIPP observes a landlord subsidy is false.

## The assistance universe, and what the published universes actually say

`ERENTSUB` applies to `ETENURE in (2,3)`. **`EVOUCHER` applies only when
`ERENTSUB = 1`.** That is its published universe, so an unassisted household has
no `EVOUCHER` answer to give: a **blank value under allocation flag 0** is the
one voucher state that universe predicts, and it is expected, not missing.

A household is `explicit_unassisted` only when `ERENTSUB = 2` carries allocation
flag `1` (reported) with no copy conflict **and** its `EVOUCHER` evidence is
exactly that expected blank. `_voucher_evidence()` grades the raw observations
rather than the collapsed measurement class, because once a value has been
discarded the class reports the flag: a populated `EVOUCHER` under flag 0 and an
empty one under flag 0 both read as `not_in_universe`, and only the second is
the expected case. The six graded states, and what each does:

| Voucher evidence | Meaning | With an explicit `ERENTSUB = 2` |
| --- | --- | --- |
| `expected_blank_not_in_universe` | blank value, flag 0 | admitted |
| `copy_conflict` | members disagree | `assistance_copy_conflict` |
| `suppressed` | flag `-99` | `assistance_universe_conflict` |
| `answer_present` | an in-domain value under any flag | `assistance_universe_conflict` |
| `applicable_without_answer` | flag 1–9, no value | `assistance_universe_conflict` |
| `unresolved` | absent column, invalid code, invalid flag | `assistance_voucher_unresolved` |

An imputed "No" is `unassisted_allocated` — its own class, excluded from the
donor universe — and conflicting voucher evidence is reported before that split,
so an imputed negative never becomes an explicit one. NIU, missing, suppressed,
allocated and conflicting statuses each keep their own class and are **never**
recoded as unassisted. A renter whose `ARENTSUB` reports flag 0 is
`assistance_not_in_universe`, which contradicts the published universe, is kept
distinct from plain missingness, and now has its own exclusion rung rather than
sharing the unknown one.

An owner is outside `ERENTSUB`'s universe. An owner with an in-universe answer
is `assistance_universe_conflict`; an owner whose assistance evidence is a copy
conflict, a suppressed flag or an applicable flag without an answer is
`owner_assistance_evidence_unresolved`, because withheld or contradictory
evidence is not the same as having no answer to give. Either way the exclusive
ladder still reports `tenure_owner`, and the class keeps the evidence visible.

Assisted households keep their voucher detail: `assisted_with_voucher`,
`assisted_without_voucher` or `assisted_voucher_unknown`.

## Exclusion ladder

`EXCLUSION_ORDER` is the declared exclusive order, so each household reports one
reason:

```
tenure_copy_conflict  tenure_unknown  tenure_owner  tenure_no_payment
assistance_universe_conflict  assistance_copy_conflict
assistance_not_in_universe  assistance_voucher_unresolved
assistance_assisted  assistance_unassisted_allocated  assistance_unknown
rent_copy_conflict  rent_unknown  weight_unknown  weight_not_positive
```

Every rung is reachable, and the test suite asserts that the exclusive counts
over an invented fixture set cover the declared order exactly.

Every household that clears the ladder is `admitted`. Nothing is trimmed by rent
magnitude, and no unassisted subset is created by dropping cheap rents. An
allocated rent amount is **not** excluded by the ladder; it is reported in
`rent_quality_admitted` so the fitting rule stays root's decision rather than
this lane's. Conditions the ladder subsumes are still counted non-exclusively —
for example `assistance_universe_conflicts_nonexclusive` counts conflicts on
owner households, which the ladder reports as `tenure_owner`.

## The version 3 predictor surface

The nine columns exist so a later unassisted contract-rent fit inherits the
published semantics of its conditioning variables rather than a summary of them.
Nothing here fits, draws, bridges a price, annualizes, logs, floors, selects a
donor mask, binds a model input or maps a recipient. `not_performed` in the
recipe lists each of those by name.

### It cannot reach the admission ladder

`EXCLUSION_ORDER` is unchanged and gains no rung. The predictor pass runs
**after** `december_housing()` has already classified every household, and takes
those households as an input; `_exclusion(tenure, assistance, rent, weight)`
takes no predictor argument, so no predictor value has a path into an admission
decision. That is a statement about the call graph, and the suite checks it two
ways: `_exclusion`'s signature is asserted directly, and a twelve-household
fixture set is measured once with these columns and once with them absent
entirely, with identical exclusion counts, assistance classes, tenure classes
and per-household exclusions required. Adding predictor rungs to the ladder
would have made the old counts something to be re-established rather than
preserved; not adding them is why they are preserved, and the test is what shows
it rather than the claim.

Prospective validity is reported on its own exclusive ladder instead:

```
predictor_copy_conflict  predictor_not_in_universe
predictor_unknown  predictor_age_coverage_incomplete
```

with `predictor_complete` for a household that clears it. The diagnostics report
the counter over all households and, separately,
`predictor_status_exclusive_admitted` over the households the measurement ladder
admitted. Neither is a fit admission: which predictor states a fit may accept is
that lane's decision, not this one's.

### State is source geography, and not fifty-one levels

`TEHC_ST` publishes **53** codes. Forty-nine states plus the District of
Columbia and — the pair a range would miss — `60` "Puerto Rico and Island Areas"
and `61` "Foreign Country", neither of which is a state or DC. The domain is
therefore the published code set, checked by strict set membership, and **not**
the range `01–56`: `03`, `07`, `14`, `43`, `52` and `57–59` are refused. A
household on `60` or `61` keeps its published code and is counted in
`state_published_non_state_area_households`; it is not folded into a state
factor, dropped, or renamed. No sub-state geography is published on this column
and none is derived: `sub_state_values_derived` is `false`.

### The state arm, now observed

`EEHC_RENTSUB`'s published universe carries the arm `TEHC_ST <= '60'`, which
version 2 could not see because the column was not carried. It is now counted in
`eehc_rentsub_state_arm` as `state_code_at_most_60`, `state_code_above_60` or
`state_unknown`. It **changes no classification**: the assistance class, the
person-month measurements and the exclusion ladder are byte-for-byte what they
are without the column, which the suite asserts by comparing a projection with
the state column against the same rows without it.

### Two household counts, and neither corrects the other

`RHNUMPER` is the publisher's own December household size; the roster count is
the number of supplied December person records for that residence. Both are
retained, and `household_size_agreement` reports `agree`, `disagree` or
`publisher_unknown` with a signed `household_size_difference`. Neither is
adjusted toward the other.

A disagreement is expected rather than defective, and the reason is a dictionary
fact: the pinned dictionary publishes `RHNUMPERWT2` (printed page 899), whose
description is the same sentence "(with Type 2 persons)". `RHNUMPER`'s own page
says nothing about Type 2 persons — the exclusion is read from the *pair*, not
from a statement on the field. The roster's treatment of Type 2 persons is not
adjudicated here either, so the disagreement is a diagnostic and never a
predictor-status failure.

`RHNUMU18` gets the same treatment, with `RHNUMU18WT2` (printed page 900) as its
paired variant. Its roster counterpart is the count of December members whose
admissible `TAGE_EHC` is under 18 — the boundary the field's own description
names. That counterpart is **withheld** (`roster_coverage_incomplete`) unless
every member carries an admissible age, rather than computed over the members
that happen to be readable.

### Income is monthly, signed, and untouched

`THTOTINC` is a December 2022 **monthly** sum whose published composition is the
dictionary's own: earnings and income of household members 15 and older plus SSI
payments received by children under 15. It is not calendar-year income and not
an ASEC `HTOTVAL` equivalent. Its published minimum is `−$10,000,000,000`, so a
negative value is an in-domain observation and is preserved signed;
`income_sign` counts `negative`, `zero`, `positive` and `unknown` separately.

The published minimum also breaks the kernel's single out-of-domain sentinel:
`_UNDOCUMENTED_SENTINEL` is `−10,000,000`, which is a perfectly legitimate
`THTOTINC` amount. `_Amount` therefore carries a per-field `sentinel`, and
`_amount_lexical` raises `SENTINEL_INSIDE_DOMAIN` rather than let a sentinel
inside a published domain silently convert a real observation into a missing
code. `THTOTINC` uses `−10,000,000,001`.

The period-consistent alternative `THCYINCPOV` (printed page 2880, status flag
`AHCYINCPOV`) is **not** carried.

### Age has no allocation quality, and a maximum needs coverage

`TAGE_EHC`'s page prints `Status Flag: NA`. There is no allocation flag, so
reported and imputed ages cannot be distinguished for this column — the one
field in the projection where that distinction is unavailable. Its measurements
classify as `observed_without_status_flag`, exactly as the topcode companions
do, and their origin is recorded as `published_source_value_without_status_flag`
so a reader cannot mistake an age for a topcode marker.

The household statistic is the **maximum over admissible December person
observations only**, with `age_coverage` reported as `complete`, `partial` or
`none` and `age_covered_members` beside it. Partial coverage produces a
**lower bound, not a maximum** — an unreadable member could be older than every
member that was read — so the measurement's status is
`household_maximum_member_age_lower_bound`, `maximum_age_is_lower_bound` is
true, and the household cannot be `predictor_complete`.

The published maximum is 90. A household maximum of 90 is counted in
`maximum_age_at_domain_maximum_households`. The page does not call it a topcode
and none is inferred: the topcode marker surface belongs to `TRENTMORT` and
`TUTILS` alone.

### What is deliberately absent

`fields_deliberately_absent` names each column a reader might expect beside
these and why it is not here: `TEHC_METRO`/`AEHC_METRO` (metro status, unused in
this increment), `RHNUMPERWT2`/`RHNUMU18WT2`/`THTOTINCT2` (the Type-2-inclusive
variants), `THCYINCPOV`/`AHCYINCPOV`, `THHLDSTATUS` (the published universe gate
of all nine columns) and `ERELRPE`/`ARELRP`. The width of the projection is a
decision, recorded as one.

## Diagnostics

`housing_diagnostics()` reports unweighted counts of: households and people;
tenure classes and raw `ETENURE`/`ERENTSUB`/`EVOUCHER`/`EPRDEBT`/`EMHDEBT` codes;
assistance classes; how every household's voucher evidence was graded, overall
and among admitted households; the exclusive exclusion ladder; per-field
evidence class, allocation quality, copy conflicts and unknown measurements;
person-month codes, quality and member disagreement; owner-debt resolution and
the evidence classes behind it; admitted rent class and quality; and boundary
counts for zero rent, domain-maximum rent, topcode-marked, partly-marked and
per-state marker households, and the utilities equivalents.

Owner debt is reported as `owner_debt_indicated`, `owner_debt_absent` or
`owner_debt_unresolved`. "Indicated" rather than "reported" is deliberate:
`EPRDEBT`/`EMHDEBT` admit an allocated answer, and *reported* means allocation
flag 1 in this vocabulary, so `owner_debt_evidence_classes` carries the
contributing evidence class for each field beside the resolution.

### Topcode markers: a zero is not a cleared non-topcode

`TUTILS_MED` and `TUTILS_STD` carry a published minimum of **$0**, so a lexical
`0` is an in-domain marker value for them, while the same token is
`invalid_domain` for the $1-minimum `TRNTMT_MED`/`TRNTMT_STD` pair. The
dictionary does not say what the public file writes for a person who is not in a
topcoded group, and **no data row has been read**, so the real incidence of zero
markers is unknown. If the file populates `0` rather than a blank,
`utilities_topcode_marked_households` would equal the household count and would
mean nothing.

The diagnostics therefore count the raw states separately — `*_topcode_marker_`
`zero`, `invalid_domain`, `blank` and `absent` `_records`, plus the household
counts for each — and **no zero is scored as a cleared non-topcode**. Read those
counts in the first-32 output before interpreting any marker total.

Cross-grain contradictions are surfaced rather than averaged away:
`person_month_receipt_in_explicit_unassisted_households`,
`person_month_receipt_in_admitted_households` and
`no_person_month_receipt_in_assisted_households`.

The only weighted block is `source_weight_totals`. It sums the household **mean
member `WPFINWGT`** and names itself exactly that, carrying
`not_a_publisher_household_sampling_weight: true` and
`not_a_rent_model_sampling_weight: true`. It exists because section 6.1 of the
requirements asks for weighted shares by tenure and assistance code; it is not a
household count. Households with an unknown weight are bucketed into
`unknown_household_weight` rather than dropped, and zero weights stay explicit —
the frozen reconciliation already expects 29 zero-weight December people.

Section 6 items this lane does **not** produce, because they need a recipient
universe and predictors that do not exist yet: common-support diagnostics
between the assisted recipient and unassisted donor universes, geographic
coverage and no-donor cell shares, and the fit/evaluation split with ACS
reported contract rent as an independent comparison target.

## Period, price and population — declared, not applied

`recipe_document()["period_and_basis_declarations"]` records all three
requirements without performing any of them. Amounts are **December 2022
monthly** in their original basis. Annualization is `not_applied` and would be a
named assumption about within-year rent stability with its own sensitivity. Any
price rebasing needs an explicit sourced rent price index, not a silent CPI-U
deflator and not the ACS `ADJHSG` housing-dollar factor, which belongs to ACS's
own reference period. SIPP's 2022 reference population is not a target year's
population, so reweighting or transfer is a separate declared step. A recipient
join must state which ASEC income cohort it targets; a 2022 SIPP charge model is
not silently paired with a 2024 ASEC cohort.

`NON_CLAIMS` records what this lane refuses to assert, including that no
landlord subsidy amount, no full contract charge for assisted households, no
program type or utility-reimbursement split is observed, that utilities are not
added into any rent quantity, that no publisher household sampling weight
exists, and that no new source byte proof is claimed — the parent's hashes and
the 39,513 / 17,646 / 17,314 / 318 / 29 reconciliation are the frozen producer's
existing evidence, reused.

## Recipe identity and refusals

The current recipe is `sipp-december-housing-projection/3`. It adds the nine
predictor columns and changes no measurement class, no assistance rule and no
rung of `EXCLUSION_ORDER`.

`SUPERSEDED_RECIPE_VERSIONS` now carries `/1` and `/2`, and the two are
superseded in different senses. Version 1's outputs never existed: no version 1
receipt was ever produced. Version 2's did — a December extraction was run under
it and reviewed — and those receipts are **not** invalidated by this bump and
are **not** reproduced by it. A version 3 run reads more columns and therefore
writes a different measurement digest and a different receipt. Version 2 remains
the identity under which the accepted measurement evidence was recorded; it is
superseded as a recipe for new extractions, not voided as a record.

Version 1 admitted an explicit `ERENTSUB = 2` whatever `EVOUCHER` carried,
folded `assistance_not_in_universe` into the unknown rung, bound no
implementation identity and copied the caller's `selection` label into the
receipt without counting the SSUIDs behind it.

`recipe_bytes()` is the canonical serialization of `recipe_document()` and
`RECIPE_SHA256` pins it. A contract change must move that pin in the same
commit; otherwise `verify_recipe()` raises `HOUSING_RECIPE_TAMPERED` before any
measurement runs. `verify_source()` raises `HOUSING_SOURCE_PINS_MISMATCH` for any
identity that is not the frozen authenticated parent, `verify_dictionary()`
raises `HOUSING_RECIPE_DICTIONARY_MISMATCH`, and `header_positions()` raises
`HOUSING_PROJECTION_HEADER` for a header that is missing, empty-named or
duplicating a declared column.

`project()` returns households, predictors, diagnostics and a
`sipp_december_housing_projection_v3` receipt carrying the recipe, its pin and
digest, the implementation binding, the source identity, the selection evidence,
the membership digest, the measurement digest, the separate predictor digest,
and aggregates only. The predictor digest is kept apart from the measurement
digest deliberately: a predictor value that moves must move the receipt, and
must not be able to look like a change in the measurement evidence the admission
ladder read.

The receipt contains no individual identifier and **no individual dollar
amount**. It is not free of every individual quantity: with a single household
in a class, `source_weight_totals` and the `known_weight_sum` in the weight
blocks expose that household's mean member `WPFINWGT`, and version 3's
`state_of_residence_codes` counter likewise exposes a lone household's state
code. No income amount and no age appear at all — the predictor diagnostics
report income only by sign and domain boundary, and age only by coverage and
domain-maximum counts. The returned measurement and predictor tuples do carry
individual amounts and identifiers, and must not be printed in a report.

### Implementation binding

`RECIPE_SHA256` pins the declared contract, not the code that applies it: a
change to `_assistance_class`, `_exclusion` or `measurement_class` can leave the
recipe bytes untouched while changing what a receipt means. The receipt
therefore carries `implementation.module_sha256` for
`sipp_housing_measurements`, `sipp_housing_predictors`, `sipp_housing_projection`
and the two frozen modules they reuse, plus a `binding` key over those digests.
The predictor module is a new code dependency of what a receipt means, so it is
inside that identity from the first version that has it, and the binding domain
moves to `sipp-december-housing-implementation-binding/2` because the bound
module set changed shape rather than only its values. It sits in the
receipt and **not** inside `recipe_bytes()`, because a module hash placed inside
the bytes it hashes cannot be computed; the recipe names what is bound and
records `inside_recipe_bytes: false`. The frozen modules are read, never
written, so the producer binding that already-published vehicle-equity receipts
carry is unchanged. No dependency version is bound here — the measurement path
is standard-library `Decimal` arithmetic under a fixed context — and the
receipt says so.

### Selection is a caller assertion

`project()` sees only the rows it is handed. `selection_evidence` records the
mode, the distinct `selected_ssuid_count`, a digest over the sorted SSUIDs,
`source_authentication: caller_asserted`, and
`relative_selection_established_by: authenticated_runner_over_the_full_december_roster`.
`first_32_ssuids` refuses more than 32 SSUIDs with `SELECTION_EXCEEDS_FIRST_32`,
but a count of 32 or fewer is **not** proof that those are the first 32 SSUIDs of
the whole source: only a runner that enumerated the full authenticated December
roster establishes that relative claim, and the receipt says so in its own words.

`source_identity` is separate from that assertion. It records the pins the
caller's `SippSource` carries, which `verify_source()` has checked against the
frozen producer's, together with
`supplied_rows_verified_against_these_bytes: false` and
`new_source_byte_proof_claimed: false`. No parent byte is opened here and the
supplied rows are not hashed, so the block says which source the caller asserts
the rows came from — never that they did.

## Extraction proposal

The runnable extraction proposal lives outside this repository, with the review
artifacts. Its current version is `us-sipp-housing-projection-extract-v2.py`
(`sipp-december-housing-extraction-proposal/2`), with
`us-sipp-housing-projection-extract-v2-selfcheck.py` beside it. Version 1,
`us-sipp-housing-projection-extract.py`, is kept unmodified as history and must
not be run.

**A version 2 extraction has been run and reviewed**, under recipe
`sipp-december-housing-projection/2`. Version 3 has not: no extraction proposal
has been written for the 38-column list, no SIPP byte has been read under it,
and every count in this document's tests comes from invented fixtures. The
counts the version 2 run produced — 17,646 December households, 4,348 admitted
and 4,057 of those with reported rent — are that run's real evidence and are
**not** predictions about a version 3 run's predictor diagnostics, which no
fixture in this repository can establish.

The proposal authenticates the parent through the frozen producer's own bounded,
no-follow, hash-checked snapshot helpers, parses the snapshot once with the
frozen record grammar, keeps every December row of the first 32 December SSUIDs,
and calls `project()`. It writes a receipt, a diagnostics JSON and a run record
to explicitly named paths under the review root and prints no source row.
`all_ssuids` remains a closed option that needs its own approval after the
first-32 result is reviewed; version 2 keeps that gate exactly and adds no new
approval flow.

Version 2 fixes what the review found in version 1:

- The **frozen reconciliation continuity check refuses before any output is
  written**. Version 1 wrote the receipt, diagnostics and run record at mode
  0400 and only then raised, so a failed run left a file named as a first-32
  receipt on disk and every rerun refused with `OUTPUT_EXISTS`.
- Every output path is checked for collision **before the parent is opened**, so
  a rerun costs nothing and can never half-write a set.
- `--no-snapshot` is removed. It hashed the original in place and never compared
  that descriptor identity at parse time, so the bytes read could differ from
  the bytes hashed. The snapshot path always applies.
- An available-memory preflight runs before the copy, using the frozen
  producer's own estimator and its 8 GiB requirement, and the copy is bracketed
  by the time and peak-RSS guard.
- Diagnostics and the run record are serialized with `canonical_json`, which
  refuses an unknown type instead of stringifying it, so `diagnostics_sha256` is
  deterministic. Version 1 used `default=str`.

The self-check runs the proposal's own code path on an invented pipe-delimited
file with a scrambled header and decoy columns on both sides. It asserts the
`all_ssuids` gate, that `--no-snapshot` no longer parses, whole-SSUID selection
over households of two and three members, a continuity refusal that leaves the
output directory empty, an `OUTPUT_EXISTS` refusal reached before a nonexistent
parent path is touched, 0400 output modes, and byte-identical canonical
diagnostics. It opens no SIPP byte. The one frozen value it replaces is
`_PINS.expected_counts`, so the continuity branch can be exercised in both
directions over a file whose counts are known; the replacement is restored and
the restoration is asserted.

## Tests

`packages/microcosm-build/tests/test_us_sipp_housing_projection.py` uses
invented rows only. It covers recipe byte identity and tampered-source refusal,
household copy disagreement, missing/NIU versus zero versus negative, owner
versus renter, assistance unknown/conflict versus explicit unassisted, every
allocation quality class, the household-copied versus person-month grain, the
topcode markers, the weights and the aggregate ladder.

For version 3 it covers: each field's re-read locator, flag, file class and
length; the 53-code state domain, the codes a range would wrongly admit, and the
two published areas that are not a state; the signed income domain, its
per-field sentinel, and a `-10000000` month that must stay a value; a sentinel
placed inside a published domain refusing rather than hiding an observation; the
absence of an age status flag and of any topcode claim at age 90; copy conflict
over a disagreeing state or income; both household counts retained through a
disagreement; the child roster counterpart withheld under incomplete age
coverage; a partial-coverage maximum age declared a lower bound; the predictor
status ladder exhausted; predictor validity reported separately for admitted
households; the exclusion ladder identical with and without the nine columns and
`_exclusion`'s signature asserted directly; a moved predictor value moving the
predictor digest while the measurement and membership digests hold; the roster
and alignment refusals; and a receipt byte-identical on a fresh-warm replay and
in a cold process that has never imported these modules.

It also covers, from the independent review: each of the four voucher states
that must not be absorbed into `explicit_unassisted` and the expected blank that
must not become missing; the discarded-value versus blank distinction in the
preserved raw observations; an owner's unresolved evidence; the
`SELECTION_EXCEEDS_FIRST_32` refusal and the receipt's selection evidence and
source identity; the implementation binding against the real module bytes, its
absence from the recipe bytes, and that a changed module digest moves the
receipt while the recipe pin stays; the zero versus invalid-domain marker
states; a November row that cannot contradict December; one SSUID with two
residences becoming two classified households; `owner_debt_absent`; an assisted
household with no member receipt; `ETENURE = 3` with a reported `ERENTSUB = 2`;
a renter whose `ARENTMORT = 0` reaches `rent_unknown`; and person-month evidence
that never reaches the household measurements.
