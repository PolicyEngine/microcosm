# US employer-sponsored insurance premiums (#454)

Status as of 2026-10-10. Code: `microcosm.build.us_runtime.esi_premiums`
(stage `meps_esi_premiums`). Receipts: `experiments/us-esi-454/`.

## What the stage produces

One PolicyEngine-US person input that no Microcosm stage produced,
`employer_sponsored_insurance_premiums`. Its consumer was read in the locked
engine, policyengine-us 2.2.1: `gov.household.cbo_market_income_additions`,
summed into `cbo_household_market_income`. No other variable or parameter
list names it.

`tests/engine_contract/us/test_us_esi_premiums_engine.py` computes it in the
engine for one worker: a $10,000 employer premium raises
`cbo_household_market_income` by exactly $10,000 and moves none of the tax
measures.

It was a recorded parity gap: the retired eCPS derivation (archived
`42ed5d45`, `datasets/cps/cps.py` L197-271) read the CPS ASEC current
employment-based coverage fields, and every pinned processed ASEC input omits
them.

The stage does not produce `pre_tax_health_insurance_premiums`, the employee
share paid by pre-tax payroll deduction. See "The pre-tax employee premium"
below.

## Source fields

`asec_census_person_columns` now restores six more Census person columns into
every pooled vintage, by exact one-to-one `PERIDNUM` identity from the pinned
official person members (`docs/us-asec-census-person-columns.md`). None of the
four pinned inputs (income years 2022 to 2025) carries any of them.

| Column | Use | Codes |
|---|---|---|
| `NOW_OWNGRP` | current employment-based policyholder | 1 yes, 2 no, 0 out of universe |
| `NOW_HIPAID` | employer paid the premium | 1 all, 2 some, 3 none, 0 out of universe |
| `NOW_GRPFTYP2` | coverage tier | 1 family, 2 self plus one, 3 self-only |
| `NOW_GRPFTYP` | cross-check on the tier | 1 family, 2 self-only |
| `PEMLR` | employed at interview | 1 at work, 2 absent |
| `NOEMP` | employer size (the work-experience item "persons who work for employer") | 1 under 10, 2 10-49, 3 50-99, 4 100-499, 5 500-999, 6 1000+, 0 not in universe |

The codes are the Census API codebooks for survey years 2023 to 2026, which
are identical across the four years for all six columns, and were checked
against the four pinned person members (`pppub23` to `pppub26`). In each file:

- a policyholder always reports a payment status and a tier, and nobody else
  reports either;
- `NOW_GRPFTYP` family equals `NOW_GRPFTYP2` family or self plus one, row for
  row.

The stage refuses any frame that breaks these two relations, carries a missing
or out-of-codebook value, or lacks a column. It never defaults a value. It
does not read `NOW_GRP` or `has_esi`: neither identifies the policyholder. (In
the four files every policyholder also has `NOW_GRP == 1`; that was observed,
and the stage does not enforce it.)

`PEIO1COW` (class of worker) is in the pinned inputs already. State is the
household `GESTFIPS`.

## Employer premium

### Two universes

The national accounts that publish an employer-contribution total count
contributions for active **and** retired employees (see the anchor section).
The stage therefore keeps two universes apart:

- **Anchor universe: every current policyholder** (`NOW_OWNGRP` 1), employed
  or not. Each is priced from MEPS-IC, and one factor scales this universe to
  the anchor.
- **Column universe: employed policyholders with an employer** (`PEMLR` 1 or
  2, `PEIO1COW` 1 to 6). Only they carry
  `employer_sponsored_insurance_premiums`, at the same factor.

Everyone else carries zero in the column:

- dependents, because the policyholder carries the premium;
- retired, unemployed and other non-employed policyholders, because their
  coverage is not compensation of a current job;
- the unincorporated self-employed and workers without pay (`PEIO1COW` 7 and
  8), whose current job is not an employer's job. Their group coverage is
  priced in the anchor universe, like a retiree's, and left out of the
  column.

The non-employed policyholders' scaled share is priced and then left out. It is
not loaded onto workers.

### Cells

MEPS-IC average total premium and average employee contribution per enrolled
employee, by coverage tier:

- Private employers (`PEIO1COW` 4 to 6): MEPS-IC 2025 Series II, by State and
  firm size. `NOEMP` 1 and 2 take the under-50 column, 3 to 6 the 50-or-more
  column, and 0 (not in universe) the State's all-sizes column. MEPS-IC's
  finer bands (10-24, 25-99, 100-999) do not nest inside the `NOEMP` bands, so
  they cannot be used.
- State government (`PEIO1COW` 2): MEPS-IC 2024 Series III, State-government
  column, by census division.
- Local and federal government (`PEIO1COW` 3 and 1): the same table's
  all-governments column. MEPS-IC does not survey the federal government; this
  is a stand-in.
- No employer class (`PEIO1COW` 0, 7 or 8): the State's all-sizes private
  column. A non-employed policyholder who does report a class of worker keeps
  that employer's cell.

`NOEMP` and `PEIO1COW` describe different jobs. `PEIO1COW` is the class of
worker of the current job. `NOEMP` is the size of the employer of the longest
job held last year. A worker who changed jobs takes the earlier employer's
size band, and a current worker with no job last year (`NOEMP` 0) takes the
State's all-sizes column.

Series III for 2025 is not yet published. The 2024 government cells are aged
to 2025 by the private-sector national ratio of the same tier and measure
(for example single premium, 9,025 / 8,486).

AHRQ suppresses nine small-firm employee-contribution cells in the 2025 State
tables (family: AK, DE, NM, SC, WV; self plus one: AK, KY, SC, SD). Each takes
the national small-firm cell times the State's all-sizes level relative to the
nation. No premium cell the stage reads is suppressed.

No MEPS-IC table prices retiree coverage, so a non-employed policyholder is
priced as an active employee. If retiree plans cost less than active plans,
the employed column is understated; the receipts carry the sensitivity.

### Share by payment status

- Employer paid all (`NOW_HIPAID` 1): the cell premium.
- Employer paid some (2): the premium minus the contribution of an employee
  who pays one.
- Employer paid none (3): zero.

MEPS-IC's average employee contribution is taken over every enrollee,
including those whose coverage required no contribution. Subtracting it from
"paid some" holders only, while "paid all" holders keep the whole premium,
would overstate the cell's employer mean. The stage divides the average
contribution by one minus the published share of enrollees with no
contribution (MEPS-IC 2025 Tables II.C.4.a, II.D.4.a and II.E.4.a, United
States row by firm size):

| Tier | All sizes | Under 50 | 50 or more |
|---|---:|---:|---:|
| Self-only | 13.0% | 35.5% | 8.7% |
| Family | 7.7% | 28.6% | 4.6% |
| Self plus one | 6.0% | 23.1% | 3.6% |

A cohort with MEPS-IC's own mix of the two groups then reproduces the cell's
published employer mean (premium minus average contribution) exactly. For the
national under-50 self-only cell, premium $9,034 and average contribution
$1,924, the unconditional rule gives $7,793 and the conditional rule the
published $7,110.

Limits of this rule:

- The no-contribution shares are national. Most State cells of those tables
  are flagged unreliable or suppressed.
- Series III publishes no such share, so government cells take the private
  50-or-more share.
- CPS respondents say "employer paid all" more often than MEPS-IC
  establishments report a zero contribution (18.7% of employed policyholders,
  against the shares above). On the pinned pools the stage's raw mean among
  employer-paying workers is therefore 3.1% above the published cell means
  (`raw_over_published_employer_mean`), before the national factor.
- MEPS-IC publishes no share of enrollees who pay the whole premium. The 7.2%
  of employed policyholders who say their employer pays nothing carry zero and
  are treated as outside the MEPS-IC averages.

### Scale

One factor scales every policyholder's share so that the weighted
anchor-universe total equals the anchor for the build year. The column is that
factor times the employed policyholders' shares.

## The anchor

| Series | CY2024 | Who it covers |
|---|---:|---|
| CMS NHE Table 24, "Employer Contribution to Private Health Insurance Premiums" | **$1,047.0B** | Private, federal and State and local employers. The NHEA methodology paper (2024, p. 29) says the estimates cover "active employees, continuation of health coverage (COBRA), and retirees" of private and State and local sponsors, from MEPS-IC, and federal "employers, employees and retirees", from OPM. |
| BEA NIPA Table 7.8 line 17, B4923C, "Private group health insurance" | $977.0B | Employer contributions as a supplement to wages and salaries. BEA's State Personal Income methodology (December 2025, paragraphs 3.18 and 3.19) says its MEPS source "covers both health insurance purchased by employers for their active and retired employees" and that the federal estimate covers "active and retired federal civilian employees". |

Both series measure the same concept, and both include retirees. The stage
scales to **NHE**, the series the issue chose and the pinned feed banks
(`cms_nhe.cy2024.esi_employer_contribution_premiums`). **BEA** is recorded
beside every anchor verdict and is not gated: it is 6.7% below NHE for 2024.

Two facts differ from what the issue recorded in July:

1. BEA revised 2024. The issue cited $1,002.9B, from the release of
   2025-09-26. The annual update published 2026-09-30 puts 2024 at $977.034B.
   The stage pins the September 2026 file by SHA-256.
2. The issue described NHE as the broader series because it includes
   retirees. BEA's does too.

NHE Table 24 ends at CY2024, so the stage refuses a 2025 build year. BEA's
2025 value does not stand in for it.

### What the choice moves

The receipts restate the default pool's column under each defensible reading.
The anchor and the treatment of non-employed policyholders are methodology
choices; the first row is what the stage builds.

| Reading | Employed column | Mean per policyholder with a positive premium |
|---|---:|---:|
| **NHE over every policyholder (as built)** | **$925.7B** | **$12,515** |
| NHE over every policyholder, 65-and-over non-employed at half price | $948.1B | $12,817 |
| BEA over every policyholder | $863.9B | $11,679 |
| NHE loaded onto employed policyholders only | $1,047.0B | $14,155 |
| BEA loaded onto employed policyholders only | $977.0B | $13,209 |

One outside check bears on it. MEPS-IC's own national rows give the
private-sector employer total for enrolled, active employees: employees times
the share in establishments that offer insurance times the share enrolled
there, times each tier's share of enrollees and its premium less average
contribution. That is $668.6B for 2024 (65.6M enrollees) and $714.4B for 2025.
The private-sector part of the as-built column is $713.2B, 6.7% above the 2024
figure; under the BEA reading it would be $665.6B, 0.4% below. The check
covers the private sector only (77% of the column) and shares the cells'
source, so it informs the choice without settling it.

## The pre-tax employee premium

`pre_tax_health_insurance_premiums` has no producer, and this stage does not
add one.

PolicyEngine-US treats the premium inputs as disjoint: a premium paid by
pre-tax payroll deduction belongs only in `pre_tax_health_insurance_premiums`,
and `health_insurance_premiums_without_medicare_part_b`,
`other_health_insurance_premiums` and `health_insurance_premiums` hold the
premiums not paid that way (PolicyEngine/policyengine-us#10046). The base
carries the whole reported premium (`PHIP_VAL`) in the second set. A producer
therefore has to move the pre-tax dollars out of the reported premium, not
copy them:

- Copying them would count the same premium twice. The engine would exclude
  it from wages and still deduct it for itemizers with medical expenses above
  the floor (for a filer with $60,000 of wages, $20,000 of other medical
  expenses and a $2,000 premium, income tax of $4,445.74 against $4,685.74).
- Moving them is only right on an engine that counts the pre-tax input where
  the reported premium was counted. policyengine-us 2.2.1 does not: its
  `spm_unit_health_insurance_premiums` adds `other_health_insurance_premiums`
  and the modeled premiums, not the pre-tax input, so the moved dollars would
  drop out of SPM medical out-of-pocket expenses. #10046 adds it.
- Moving about $180 billion out of two premium inputs also changes what the
  release tool's input-mass parity gate and the eCPS parity reference expect
  of those columns.

An earlier round of this work assigned the reported premium, capped at wages,
to eligible workers at MEPS-IC pretax-offer rates and measured $180.1 billion
over 52.5 million workers on the default pool. That code is on branch
`esi-pre-tax-premiums-454` and is the starting point for the follow-up, which
is sequenced after an engine pin that includes #10046.

## Measurements on the pinned inputs

`experiments/us-esi-454/run_stage_on_pinned_pool.py` pools the real pinned
inputs the way the base build does and runs the stage. These are
pre-calibration figures at pooled ASEC weights, target year 2024.

| | 2023-2025 (default) | 2022-2024 |
|---|---:|---:|
| Current ESI policyholders | 92.4M | 92.9M |
| Employed, with an employer | 79.7M | 79.8M |
| With a positive employer premium | 74.0M | 74.1M |
| Other policyholders (priced, outside the column) | 12.7M | 13.2M |
| Raw MEPS-IC total, every policyholder | $1,006.4B | $1,012.1B |
| Scale factor | 1.040 | 1.034 |
| Anchor-universe total | $1,047.0B | $1,047.0B |
| **Employer premium column** | **$925.7B** | $923.9B |
| Employed share of the anchor universe | 88.4% | 88.2% |
| Other policyholders' share, not in the column | $121.3B | $123.1B |
| Mean per policyholder with a positive premium | **$12,515** | $12,476 |
| Mean per employed policyholder | $11,616 | $11,579 |
| By tier: family / self plus one / self-only | $393.1B / $198.2B / $334.4B | $393.2B / $199.7B / $331.0B |
| By employer: private / State and local / federal | $713.2B / $174.3B / $38.1B | $711.2B / $174.1B / $38.6B |

Of the 12.7M other policyholders in the default pool, 7.1M are retired
(`PEMLR` 5), 2.1M are otherwise out of the labor force (7), 0.9M are disabled
(6), 1.0M are unemployed (3 and 4), 1.5M are employed without an employer
class, and 0.1M have no labor-force status (0).

Both the signal gate and the anchor gate pass on each pool, before and after
the PUF-support clone, and the clone conserves both totals.

## Gates

| Gate | Where | Checks |
|---|---|---|
| `us_esi_premiums_signal_gate` | base build, before and after the clone; release tool, on the base and on the calibrated export | Column present, finite, nonnegative and non-constant; support clones agree; the weighted share of people with a positive value is plausible; no employer premium outside the column universe or where the employer pays none; every premium is one common multiple of its MEPS-IC share. |
| `us_esi_premiums_anchor_gate` | release tool, on the calibrated export, for both the dense and the sparse default | The column absent or zero-mass is red. The anchor-universe total, recomputed at the release's weights, must be within **5%** of NHE Table 24 for the release period. Employed policyholders must carry 80% to 95% of it. |
| `us_release_input_coverage_gate` | release tool | The input is `required` with no reviewed exclusion, so an export that drops or flattens it fails. |
| `assert_required_us_release_source_columns` | the L0/refit export (`l0_refit_export.export_us_l0_refit_h5`, run by `tools/export_us_l0_refit_h5.py`), unless `--allow-missing-source-columns` is passed | The column must be present and non-constant. |
| reform-coverage smoke | release tool, on the written file | Neutralizing the employer premium must lower `cbo_household_market_income` by at least $500B. |

What the gates need, and what they do not catch:

- **They need the raw ASEC columns.** Both gates recompute the assignment
  from `NOW_OWNGRP`, `NOW_HIPAID`, `NOW_GRPFTYP`, `NOW_GRPFTYP2`, `PEMLR`,
  `NOEMP`, `PEIO1COW` and State. A frame missing any of them fails both: it
  cannot be certified. Published release files carry raw ASEC person fields
  today (`PEIO1COW`, `PHIP_VAL`, `WSAL_VAL`, `NOW_MCAID`); a release path
  that dropped the restored columns would turn both gates red.
- **The raw-mean band catches unit errors only.** The signal gate requires
  the unscaled share per holder to sit between $6k and $20k, which catches a
  cell table in the wrong units. A cell table that is 20% low passes: the
  scale factor absorbs it. A re-pinned table is checked against the PDFs by
  `tools/build_us_meps_ic_esi_cells.py --check`, not by this gate.
- **The 5% tolerance is a chosen criterion.** No release has calibrated this
  column yet, so the tolerance is not derived from measured drift. The first
  release build records the drift in `esi_premiums.json`.

The stage pins the pre-calibration total exactly. Nothing calibrates the
column, so the anchor gate is what holds a release's total after reweighting
and sparse selection. `esi_premiums.json` is required release evidence: it
carries the anchor verdict, both cross-checks and the base and export signal
verdicts, and both manifests record it. `--allow-esi-premium-gaps` stops the
gates failing a diagnostic build, for example on a base built before this
stage; the failures it waives are still written to that file.

## Lineage

- Census person members: pinned in
  `education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES` (archive
  and member SHA-256), read by the #720 restoration.
- MEPS-IC cells: three AHRQ PDFs pinned by URL, byte length and SHA-256 in
  `tools/build_us_meps_ic_esi_cells.py`, extracted to
  `us_runtime/data/meps_ic_esi_premium_cells.json`, whose own SHA-256 the
  stage pins and checks on load.
- NHE and BEA files: URL, byte length and SHA-256 in the stage manifest
  (`us/source_stages.json`), and URL and SHA-256 in the module. The NHE
  digest is of the zip archive, not of the Table 24 workbook inside it.
- The base summary records the stage's full summary twice
  (`esi_premiums_signal`, `esi_premiums_post_clone_signal`): scale factor, raw
  and scaled totals for both universes, totals by tier and employer sector,
  weighted policyholder counts, the anchor, the cross-check and the cell-table
  digest.

## What this does not do

- **The multispine pool.** `multispine_pool.py` runs its own operator chain on
  CPS-source rows that carry only part of the stacked population's weight, so
  scaling to a national total there needs its own design. Until then the
  coverage and anchor gates are red on that lineage. The raw-stage operator
  boundary already refuses a source frame that carries the output.
- **ACS local releases.** An ACS spine pooled beside a donor that carries the
  input has it null on its ACS rows. `fill_reviewed_nulls` refuses that frame
  instead of default-filling it to zero, so this lineage does not build until
  the input is assigned or transferred for every spine.
- **A calibration target.** The column is the employed part of the NHE fact,
  so the fact cannot be compiled as a sum target on it. The target-parity
  manifest keeps the two `cms_nhe` ESI families as `deferred` reviewed
  exclusions and records BEA's series as a source-absent family.
- **The standalone L0/refit export.** `tools/export_us_l0_refit_h5.py`
  writes an H5 without running the signal or anchor gate. By default it
  refuses a frame whose column is absent or constant, and nothing more: the
  two ESI gates run only in the base build and in
  `tools/build_us_fiscal_refresh_release.py`.
- **The pre-tax employee premium.** See the section above.
