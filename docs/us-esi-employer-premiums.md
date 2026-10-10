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
  8), who have no employer to pay a share.

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
- No employer class (`PEIO1COW` 0, 7 or 8, which is most non-employed
  policyholders): the State's all-sizes private column. A non-employed
  policyholder who does report a class of worker keeps that employer's cell.

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

## ACS local releases

The ACS local lane (`tools/build_us_acs_multispine_base.py`, then
`tools/build_us_acs_local_release.py`) pools a dense donor release with the
ACS 2024 1-year spine. `with_optional_acs_spine` multiplies every donor
household weight by `1 - acs_share` and gives ACS the rest of the donor's
household mass (half each by default). The donor rows arrive with the column
and the raw coverage columns the release carries. ACS rows have neither.

### The lane transfers the column in its own pass

`acs_local_esi_premium_transfer_target_families` is a one-family plan,
`source_operator_esi_premiums`, that only the lane's staging build uses, as it
does for usual hours (#765). The declared ACS plan
(`declared_acs_transfer_target_families`) does not gain the column:

- The lane's main transfer fits every declared family on the PUF-detail role
  (`ACS_DONOR_CHANNEL_AUTO`), the wrong donor for this column (next section).
- Under one of the declared plan's existing families, the column would also
  leave the stacked pool's `source_operator_esi_premiums` family:
  `pool_transfer_target_families` adds a `source_operator_*` family only for
  outputs the declared plan does not own. And it would enter the frozen v11
  authority reconstruction (`stacked_spine._legacy_stacked_authority_receipt`),
  which leaves out post-v11 targets by family name; scoring an attested
  schema-9 pool refuses an authority that differs from that reconstruction.
  Declaring it under its own `source_operator_esi_premiums` family would avoid
  both, but not the donor problem.
- The declared plan is also the default registry of the spine-agreement
  battery (`spine_agreement`), which would gain the column as a metric.

Taking the lane's ACS rows from the stacked pool instead would replace the
lane. Its two tools read the legacy spine tags, the legacy support roles and
the staging summary's null register, and its hours repair refuses an assembled
frame.

### The donor rows are the ASEC observations

The transfer fits on the donor's ASEC observation role (support clone 0). The
lane's tax-detail families fit on the PUF-detail role, which is the wrong
donor for this column:

- On the observation role, `employment_income_before_lsr` is the measured CPS
  wage, and a person has a premium only if they hold a job with an employer.
- PUF support imputation writes the clone's wages
  (`PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS`), while the clone's premium is a
  copy of its source person's. On those rows the premium no longer follows the
  wage.

The recipient side is the measured ACS wage (`WAGP`), so both sides of the fit
see a measured wage, as in the stacked pool's early fill. On the test fixture,
where only wages tell the employed from the rest, a fit on ASEC observations
gives no ACS person without wages a premium; a fit on the PUF-detail role
gives one to more than a tenth of them and to fewer workers
(`test_the_puf_detail_role_would_put_premiums_on_people_without_wages`).

`prepare_acs_local_esi_premium_donor` qualifies the donor before the ACS
sources are fetched. It refuses a donor without the column, the raw coverage
columns, wages or legacy support roles, and one that fails the stage's signal
gate.

### The ACS rows are held to the donor rows' scale

Nothing in a QRF draw keeps a total. Once the frame is pooled,
`with_acs_local_esi_premium_anchor` runs the stacked pool's kernel,
`with_us_esi_premium_pool_anchor`, on it. On the ACS rows it:

1. clears the premium where the person reports no wages; then
2. scales the rest by one factor so the ACS rows carry the same weighted
   premium per unit of household mass as the donor rows.

It never rewrites a donor row. These hold for every lane frame
(`tests/engine_free/us/test_us_acs_local_esi_premiums.py`, property-tested
over the share, both spines' weights and the transfer's draws):

- the lane-wide weighted column equals the donor rows' total divided by their
  share of household mass, `1 - acs_share`. Pooling multiplied every donor
  household weight by that share, so the lane-wide column is the donor
  release's own column;
- both spines carry the same weighted premium per unit of household mass, so
  moving mass from one spine to the other does not move the total;
- an ACS person without wages carries nothing;
- an anchored frame is returned unchanged.

The lane runs the kernel and both gates on a narrow view of the frame, for two
reasons:

- **Source ids.** The kernel finds a support clone's wages by
  `person_source_id` across the whole frame. The stacked pool makes that id
  unique at assembly. This lane does not: ACS rows keep their ids from before
  the pooling remap as source ids, and those can repeat a donor's. Every ACS
  row here is its own source record, so the view leaves the lookup's columns
  out and the kernel reads each row's own wages. A transferred row that is not
  clone 0 is refused.
- **Memory.** The kernel copies every table it is given and each gate copies
  the person table. The staging frame holds two full spines, and the view
  carries only the columns they read.

**Assumption.** As in the stacked pool, ACS rows have no policyholder flag, so
the lane's anchor-universe total is the donor rows' total plus the ACS column
over the donor rows' employed share of the anchor universe.

The donor is a calibrated release. Its anchor-universe total is within the
release tolerance of the anchor, not on it, and the staging frame starts from
the same place.

### What the lane records

- **Staging.** Both gates run on the pooled frame before the staging H5 is
  written. A red gate fails the build. The verdicts, the donor receipt and the
  anchor receipt are in the staging summary under `local_esi_premiums`.
- **Finalize.** Both gates run on the calibrated artifact. The gate report
  carries them as `esi_premiums_signal` and `esi_premiums_anchor`, each bound
  to the artifact's SHA-256, and either one red fails finalize. The report's
  `input_coverage` entry used to be a fixed pass; it now passes only if both
  gates do.
- **Package.** A report without both verdicts, with a verdict bound to other
  bytes, or with a red verdict is refused before a release directory exists.
- **`fill_reviewed_nulls`** still refuses a frame whose column is null on some
  rows. A current staging frame has none; a staging H5 built before this
  change does.

`--allow-esi-premium-gaps`, on both tools, records a red verdict without
failing a diagnostic build from a donor that predates the stage. No row then
carries the column.

No certified donor release carries the column yet, so the lane's fill has not
been measured on real inputs. The first staging build records it in
`local_esi_premiums`.

## Gates

| Gate | Where | Checks |
|---|---|---|
| `us_esi_premiums_signal_gate` | base build, before and after the clone; release tool, on the base and on the calibrated export; ACS local lane, on the staging frame and the calibrated artifact | Column present, finite, nonnegative and non-constant; support clones agree; the weighted share of people with a positive value is plausible; no employer premium outside the column universe or where the employer pays none; every premium is one common multiple of its MEPS-IC share. |
| `us_esi_premiums_anchor_gate` | release tool, on the calibrated export, for both the dense and the sparse default; ACS local lane, on the staging frame and the calibrated artifact | The column absent or zero-mass is red. The anchor-universe total, recomputed at the release's weights, must be within **5%** of NHE Table 24 for the release period. Employed policyholders must carry 80% to 95% of it. |
| `us_release_input_coverage_gate` | release tool | The input is `required` with no reviewed exclusion, so an export that drops or flattens it fails. |
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
  (`us/source_stages.json`), and URL and SHA-256 in the module.
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
- **An ACS local release from an older donor.** The lane transfers the column
  from its donor release. A donor built before the stage has nothing to
  transfer, and the lane refuses it unless a diagnostic build waives the
  gates.
- **ACS-native predictors for the lane's fill.** The lane's fill conditions on
  age, sex, State and income. It does not use ACS's own employer-coverage,
  class-of-worker or employment-status items.
- **A second verdict on the ACS rows' cells.** The gates recompute the MEPS-IC
  assignment on the donor rows. They compare the ACS rows with those rows and
  cannot check them against the cells.
- **A calibration target.** The column is the employed part of the NHE fact,
  so the fact cannot be compiled as a sum target on it. The target-parity
  manifest keeps the two `cms_nhe` ESI families as `deferred` reviewed
  exclusions and records BEA's series as a source-absent family.
- **The pre-tax employee premium.** See the section above.
