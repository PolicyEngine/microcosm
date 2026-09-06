# December SIPP housing source projection

`microcosm.build.sipp_housing_measurements` measures December housing evidence
from explicitly supplied 2023 SIPP rows, and
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
ordinary viewer shows P+1. Column presence was confirmed against the pinned
parent's pipe-delimited header only. No SIPP data row was read to write this.

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

## The 29-column projection

```
SSUID PNUM MONTHCODE ERESIDENCEID WPFINWGT
ETENURE ATENURE TRENTMORT ARENTMORT TUTILS AUTILS
ERENTSUB ARENTSUB EVOUCHER AVOUCHER
EPRDEBT APRDEBT EMHDEBT AMHDEBT
EEHC_TEN AEHC_TEN EEHC_RENTSUB AEHC_RENTSUB EEHC_VOUCHER AEHC_VOUCHER
TRNTMT_MED TRNTMT_STD TUTILS_MED TUTILS_STD
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
plain missingness at that grain.

## Grain

`ETENURE`, `TRENTMORT`, `TUTILS`, `ERENTSUB`, `EVOUCHER`, `EPRDEBT` and
`EMHDEBT` are household-level values copied to every member; they go through the
frozen kernel's copy-agreement rule, and a within-household disagreement is
`household_copy_conflict` with an unknown measurement, exactly as the vehicle
fields behave. `EEHC_TEN`, `EEHC_RENTSUB` and `EEHC_VOUCHER` are person-month
values that are **never** copied: members may legitimately differ, which is
reported as a member disagreement rather than a conflict, and they never set the
household's assistance class. The topcode companions are person-level.

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
no `EVOUCHER` answer to give and its blank is expected, not missing. The
applicable household indicator for an unassisted determination is therefore
`ERENTSUB` alone, and an applicable `EVOUCHER` answer alongside an explicit
`ERENTSUB = 2` is recorded as `assistance_universe_conflict` rather than
absorbed.

A household is `explicit_unassisted` only when `ERENTSUB = 2` carries allocation
flag `1` (reported) with no copy conflict. An imputed "No" is
`unassisted_allocated` — its own class, excluded from the donor universe. NIU,
missing, suppressed, allocated and conflicting statuses each keep their own
class and are **never** recoded as unassisted. A renter whose `ARENTSUB` reports
flag 0 is `assistance_not_in_universe`, which contradicts the published universe
and is kept distinct from plain missingness.

Assisted households keep their voucher detail: `assisted_with_voucher`,
`assisted_without_voucher` or `assisted_voucher_unknown`.

## Exclusion ladder

`EXCLUSION_ORDER` is the declared exclusive order, so each household reports one
reason:

```
tenure_copy_conflict  tenure_unknown  tenure_owner  tenure_no_payment
assistance_universe_conflict  assistance_copy_conflict  assistance_assisted
assistance_unassisted_allocated  assistance_unknown
rent_copy_conflict  rent_unknown  weight_unknown  weight_not_positive
```

Every household that clears the ladder is `admitted`. Nothing is trimmed by rent
magnitude, and no unassisted subset is created by dropping cheap rents. An
allocated rent amount is **not** excluded by the ladder; it is reported in
`rent_quality_admitted` so the fitting rule stays root's decision rather than
this lane's. Conditions the ladder subsumes are still counted non-exclusively —
for example `assistance_universe_conflicts_nonexclusive` counts conflicts on
owner households, which the ladder reports as `tenure_owner`.

## Diagnostics

`housing_diagnostics()` reports unweighted counts of: households and people;
tenure classes and raw `ETENURE`/`ERENTSUB`/`EVOUCHER`/`EPRDEBT`/`EMHDEBT` codes;
assistance classes; the exclusive exclusion ladder; per-field evidence class,
allocation quality, copy conflicts and unknown measurements; person-month codes,
quality and member disagreement; owner-debt resolution; admitted rent class and
quality; and boundary counts for zero rent, domain-maximum rent, topcode-marked
and partly-marked households, and the utilities equivalents.

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

`recipe_bytes()` is the canonical serialization of `recipe_document()` and
`RECIPE_SHA256` pins it. A contract change must move that pin in the same
commit; otherwise `verify_recipe()` raises `HOUSING_RECIPE_TAMPERED` before any
measurement runs. `verify_source()` raises `HOUSING_SOURCE_PINS_MISMATCH` for any
identity that is not the frozen authenticated parent, `verify_dictionary()`
raises `HOUSING_RECIPE_DICTIONARY_MISMATCH`, and `header_positions()` raises
`HOUSING_PROJECTION_HEADER` for a header that is missing, empty-named or
duplicating a declared column.

`project()` returns households, diagnostics and a
`sipp_december_housing_projection_v1` receipt carrying the recipe, its pin and
digest, the membership and measurement digests, and aggregates only. The receipt
contains no individual identifier and no individual amount; the returned
measurement tuples necessarily do, and must not be printed in a report.

## Extraction proposal

The runnable first-32-SSUID proposal lives outside this repository, with the
review artifacts, and has **not** been executed. It authenticates the parent
through the frozen producer's own bounded, no-follow, hash-checked snapshot
helpers, parses the snapshot once with the frozen record grammar, keeps December
rows for the first 32 December SSUIDs, and calls `project()`. It writes a
receipt and a diagnostics JSON to explicitly named paths under the review root
and prints no source row. `all_ssuids` remains a closed option that needs its
own approval after the first-32 result is reviewed; it is not implied here.

## Tests

`packages/microcosm-build/tests/test_us_sipp_housing_projection.py` uses
invented rows only. It covers recipe byte identity and tampered-source refusal,
household copy disagreement, missing/NIU versus zero versus negative, owner
versus renter, assistance unknown/conflict versus explicit unassisted, every
allocation quality class, the household-copied versus person-month grain, the
topcode markers, the weights and the aggregate ladder.
