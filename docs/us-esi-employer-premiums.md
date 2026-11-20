# US employer-sponsored insurance premiums (#454)

Status as of 2026-11-18. Code: `microcosm.build.us_runtime.esi_premiums`
(stage `meps_esi_premiums`) and, for ACS local releases,
`microcosm.build.us_runtime.acs_local_esi_premiums`. Receipts:
`experiments/us-esi-454/`.

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
job held last year. A worker whose longest job last year was with a different
employer from the current one takes that other employer's size band, and a
current worker with no job last year (`NOEMP` 0) takes the State's all-sizes
column.

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
| BEA NIPA Table 7.8 line 17, B4923C, "Private group health insurance" | **$977.0B** | Employer contributions as a supplement to wages and salaries. BEA's State Personal Income methodology (December 2025, paragraphs 3.18 and 3.19) says its MEPS source "covers both health insurance purchased by employers for their active and retired employees" and that the federal estimate covers "active and retired federal civilian employees". |
| CMS NHE Table 24, "Employer Contribution to Private Health Insurance Premiums" | $1,047.0B | Private, federal and State and local employers. The NHEA methodology paper (2024, p. 29) says the estimates cover "active employees, continuation of health coverage (COBRA), and retirees" of private and State and local sponsors, from MEPS-IC, and federal "employers, employees and retirees", from OPM. |

Both series measure the same concept, and both include retirees. The stage
scales to **BEA**, as decision d1317 (2026-10-11) ruled. **NHE** is recorded
beside every anchor verdict and is not gated: it is 7.2% above BEA for 2024,
more than the anchor gate's tolerance, so a release scaled to NHE would fail
that gate.

The stage pins BEA's values itself (`EMPLOYER_PREMIUM_ANCHOR`, the
Section 7 workbook by URL and SHA-256). The pinned US Chronicle feed banks the
same series (record sets `bea_nipa.cy<year>.private_group_health_insurance`,
PolicyEngine/chronicle#331; `docs/us-chronicle-feed-repin.md`). Two checks
hold them together. A test compares the anchor with the three facts the
committed re-pin report says the pinned feed added, so changing one without
the other fails CI. And the release tool refuses a feed whose values differ
from the stage's (`us_esi_premiums_anchor_feed_mismatches`); a diagnostic run
under `--allow-unpinned-feed` or `--allow-esi-premium-gaps` continues and
records the difference in `esi_premiums.json`, or in the report of a
`--dry-run-gates-report` run, which stops before that file is written.

Two facts differ from what #454 recorded in July:

1. BEA revised 2024. The issue cited $1,002.9B, from the release of
   2025-09-26. The annual update published 2026-09-30 puts 2024 at $977.034B
   (2023 $923.195B, unchanged; 2025 $1,027.929B).
2. The issue described NHE as the broader series because it includes
   retirees. BEA's does too.

The workbook column covers 2023 to 2025, so the stage refuses any other build
year.

### What the choice moves

The receipts restate the default pool's column under each defensible reading.
The anchor and the treatment of non-employed policyholders are methodology
choices; the first row is what the stage builds.

| Reading | Employed column | Mean per policyholder with a positive premium |
|---|---:|---:|
| **BEA over every policyholder (as built, d1317)** | **$863.9B** | **$11,679** |
| BEA over every policyholder, 65-and-over non-employed at half price | $884.7B | $11,961 |
| NHE over every policyholder | $925.7B | $12,515 |
| BEA loaded onto employed policyholders only | $977.0B | $13,209 |
| NHE loaded onto employed policyholders only | $1,047.0B | $14,155 |

One outside check bears on it, and decided it. MEPS-IC's own national rows
give the private-sector employer total for enrolled, active employees:
employees times the share in establishments that offer insurance times the
share enrolled there, times each tier's share of enrollees and its premium
less average contribution. That is $668.6B for 2024 (65.6M enrollees) and
$714.4B for 2025. The private-sector part of the as-built column is $665.6B,
0.4% below the 2024 figure; under the NHE reading it would be $713.2B, 6.7%
above. The check covers the private sector only (77% of the column) and
shares the cells' source.

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
| Scale factor | 0.971 | 0.965 |
| Anchor-universe total | $977.0B | $977.0B |
| **Employer premium column** | **$863.9B** | $862.2B |
| Employed share of the anchor universe | 88.4% | 88.2% |
| Other policyholders' share, not in the column | $113.2B | $114.9B |
| Mean per policyholder with a positive premium | **$11,679** | $11,642 |
| Mean per employed policyholder | $10,840 | $10,805 |
| By tier: family / self plus one / self-only | $366.9B / $184.9B / $312.1B | $366.9B / $186.3B / $308.9B |
| By employer: private / State and local / federal | $665.6B / $162.7B / $35.6B | $663.7B / $162.5B / $36.0B |

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
`microcosm.build.us_runtime.acs_local_esi_premiums` puts the column on them:
it qualifies the donor, transfers the column and holds the ACS rows to the
donor rows' scale.

### The lane transfers the column in its own pass

`acs_local_esi_premium_transfer_target_families` is a one-family plan,
`source_operator_esi_premiums`, that only the lane's staging build uses, as it
does for usual hours (#765) and the ASEC income families (#1022). The pass
runs after those two and immediately before the lane's shared transfer, so the
hours and income passes see the frame they would see without it. The declared
ACS plan (`declared_acs_transfer_target_families`) does not gain the column:

- The shared transfer fits every declared family on the PUF-detail role by
  default (`--donor-channel auto`), the wrong donor for this column (next
  section).
- Other plans start from the declared one: the multispine pool's transfer plan
  (`pool_transfer_target_families`) and the spine-agreement battery's default
  registry (`spine_agreement`). A column declared there would reach both.

The transfer's predictors are fixed lists (`ACS_PERSON_TRANSFER_PREDICTORS`,
`ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS`) that do not include the premium, so
the premium the pass writes cannot enter another family's fit. On the test
fixture, the shared transfer draws the same values with and without the pass,
every other column of every table is byte-equal, and the hours and income
passes make the same calls and draws
(`test_the_esi_pass_moves_no_other_value_on_any_row`,
`test_the_esi_pass_changes_no_call_or_draw_of_the_other_passes`).

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
see a measured wage. On the test fixture, where only wages tell the employed
from the rest, a fit on ASEC observations gives no ACS person without wages a
premium; a fit on the PUF-detail role gives one to more than a tenth of them
and to fewer workers
(`test_the_puf_detail_role_would_put_premiums_on_people_without_wages`).

`prepare_acs_local_esi_premium_donor` qualifies the donor before the staging
build loads the SIPP donor or fetches the ACS sources. It refuses a donor
without the column, the raw coverage columns, the CPS record id (`PERIDNUM`),
wages or legacy support roles, one with a row that lacks any raw coverage
column or a record id, one whose ASEC observations have a missing wage, and
one that fails the stage's signal gate. Every donor row is a CPS record, so a
row that lost those fields stops the build here, in minutes, not after the
transfer, even one that lost both: the signal gate grades the donor as the
pooled frame will be graded (see "The gates grade the lane's frame by row
kind") and would read such a row as a transferred one.

### The ACS rows are held to the donor rows' scale

Nothing in a QRF draw keeps a total. Once the frame is pooled,
`with_acs_local_esi_premium_anchor` runs the mixed-frame kernel in
`esi_premiums`, `with_us_esi_premium_pool_anchor`, on it. The kernel tells the
donor rows from the ACS rows by the CPS record id and the raw coverage
columns, which only the donor rows carry. On the ACS rows it:

1. clears the premium where the person reports zero wages; then
2. scales the rest by one factor so the ACS rows carry the same weighted
   premium per unit of household mass as the donor rows.

The kernel refuses a transferred row whose wage is missing: a missing wage is
not a zero wage. ACS PUMS leaves `WAGP` blank below age 15, outside its
earnings universe, and the lane keeps that blank in the staging frame; the
release tool's reviewed null fill (`fill_reviewed_nulls`) sets it to zero for
the engine. The anchor reads those people's wages as zero under the universe
rule `acs_income_universe` names (`acs_2024_pums_wagp_age_15_plus`), in its
own view of the frame, and writes no zero into the frame. The rest of the rule
stops the build:

- a blank raw `WAGP` at age 15 or older, or at an unknown age, however the
  mapped wage was filled;
- below 15, a raw `WAGP` that is not blank, or a mapped wage that is neither
  null nor zero.

The kernel resolves the same rule itself
(`resolve_acs_pums_earnings_universe`), but only on rows whose support channel
is `acs`. This lane's ACS rows carry the late spine's channel
(`acs_2024_1yr`), so the kernel's check sees none of them and its receipt
counts no rows. The lane applies the rule before calling the kernel, and its
receipt (`wage_universe`) counts what it checked.

The anchor never rewrites a donor row. These hold for every lane frame
(`tests/engine_free/us/test_us_acs_local_esi_premiums.py`, property-tested
over the ACS share, both spines' weights and the transferred premiums and
wages):

- the lane-wide weighted column equals the donor rows' total divided by their
  share of household mass, `1 - acs_share`. Pooling multiplied every donor
  household weight by that share, so the lane-wide column is the donor
  release's own column;
- both spines carry the same weighted premium per unit of household mass, so
  moving mass from one spine to the other does not move the total;
- an ACS person with zero wages, or below age 15, carries nothing;
- the frame's own wage blanks are never overwritten;
- an anchored frame is returned unchanged.

The lane runs the kernel and both gates on a narrow view of the frame, for two
reasons:

- **Source ids.** The kernel finds a support clone's wages and draw by
  `person_source_id` across the whole frame, so it needs that id unique there.
  This lane's is not: ACS rows keep their ids from before the pooling remap as
  source ids, and those can repeat a donor's. Every ACS row here is its own
  source record, so the view leaves the source id out and the kernel reads
  each row's own wages (`test_acs_ids_that_repeat_the_donors_change_nothing`).
  The view keeps the clone index, which the kernel's earnings universe
  requires, and the lane refuses a transferred row that is not clone 0.
- **Memory.** The kernel copies every table it is given and each gate copies
  the person table. The staging frame holds two full spines, and the view
  carries only the columns they read.

**Assumption.** ACS rows have no policyholder flag, so the lane's
anchor-universe total is the donor rows' total plus the ACS column over the
donor rows' employed share of the anchor universe. That assumes employed
policyholders carry the same share of employer premiums on the ACS rows as on
the donor rows.

The donor is a calibrated release. Its anchor-universe total is within the
release tolerance of the anchor, not on it, and the staging frame starts from
the same place.

### The gates grade the lane's frame by row kind

`us_esi_premiums_summary` and both gates take `allow_transferred_rows`,
`False` by default. The lane passes `True` everywhere it grades: in the donor
qualifier's signal gate and in `acs_local_esi_premium_signal_gate` and
`acs_local_esi_premium_anchor_gate`. The staging build calls the two through
`acs_local_esi_premium_gates`; finalize calls them one by one, each beside its
entry of the gate battery's shadow. The anchor kernel applies the same rule
itself. The donor rows then get the full MEPS-IC recomputation, and the ACS
rows are compared with them ("What the gates need", under "Gates", has the
rule and its refusals).

The base build (`tools/build_us_puf_support_base.py`), the release tool
(`tools/build_us_fiscal_refresh_release.py`) and the stage itself, when it
re-checks a frame that already carries the column, keep the default. Under it
every row must carry every raw coverage column, a row without them fails both
gates with each missing column and its row count named, and the CPS record id
is not read. The lane's frame is red under the default
(`test_the_lane_frame_is_red_under_route_as_default_rule`), and Route A's
callers grade as they did before a mixed frame could be graded.
`tests/engine_free/us/test_us_esi_premiums_route_a_differential.py` checks
this against a vendored copy of the earlier gates
(`test_support/microcosm_build/us_esi_premiums_main_oracle.py`): on Route
A-shaped frames every verdict agrees, and where the copy grades a frame
without refusing it, the failures and every recorded figure agree bit for bit.

### What the lane records

- **Staging.** The donor is qualified before the SIPP donor and the ACS
  sources are loaded. Both gates run on the frame the staging H5 is written
  from, after the last stage that replaces it. A red gate fails the build
  before the H5 is written. The verdicts, the donor receipt and the anchor
  receipt are in the staging summary under `local_esi_premiums`.
- **Finalize.** Both gates run on the calibrated artifact. The gate report
  carries them as `esi_premiums_signal` and `esi_premiums_anchor`, each bound
  to the artifact's SHA-256, and either one red fails finalize once the report
  is written. The report's `input_coverage` entry used to be a fixed pass; it
  now passes only if both gates do. Each gate is also a `us_local` entry of
  `us/gates.json` (`us_local_esi_premiums_signal`,
  `us_local_esi_premiums_anchor`), bound to the lane's row-kind gates, so the
  gate battery's shadow (microcosm#1585 item 11) evaluates it beside today's
  call and compares the verdicts.
- **Package.** A report without both verdicts, with a verdict bound to other
  bytes, or with a red verdict is refused before a release directory exists.
- **`fill_reviewed_nulls`** still refuses a frame whose column is null on some
  rows. A current staging frame has none. A staging H5 built before this
  change has them, and so does one whose donor qualification a diagnostic
  build waived.

`--allow-esi-premium-gaps`, on both tools, is for diagnostic builds only, for
example from a donor that predates the stage. In staging it turns a refused
donor and red gates into records in `local_esi_premiums` instead of failures.
From a refused donor the lane transfers nothing: if the donor lacks the
column, no row carries it; if it carries the column but fails qualification,
the ACS rows stay null and `fill_reviewed_nulls` refuses the frame. In
finalize the flag records a red verdict without failing, and the gate entry
keeps the failures with `enforced: false`. Package takes a red verdict only as
that recorded waiver: with the flag it still refuses a red verdict finalize
enforced. The gate battery's shadow records the waiver on each ESI entry as
the finalize run that evaluated it had it, so a package run with other flags
does not rewrite it. Finalize keeps that record beside its shadow
observations; a finalize checkpoint saved before it did has observations and
no record, and the shadow then records the ESI entries' waiver as unknown
rather than as none. A release packaged that way still fails the
local-area release contract (`microcosm.data.contract`), which accepts only
gates that passed. A Modal plan can set the waiver for the release tool
(`allow_esi_premium_gaps`) but not for staging: the staging runner owns the
flag and never passes it, so every Modal staging run qualifies the donor and
enforces both gates.

No ACS local build has run this pass on real inputs yet, so the lane's fill
is unmeasured. The first staging build records it in `local_esi_premiums`.
That build's finalize is also the first grading of both gates at calibrated
weights: calibration reweights each household, so the anchor-universe total
and the ratio between the spines can move after staging, and these gates are
what hold them.

## Gates

| Gate | Where | Checks |
|---|---|---|
| `us_esi_premiums_signal_gate` | base build, before and after the clone; release tool, on the base and on the calibrated export; ACS local lane, on the donor release, the staging frame and the calibrated artifact | Column present, finite, nonnegative and non-constant; support clones agree; the weighted share of people with a positive value is plausible; no employer premium outside the column universe or where the employer pays none; every premium is one common multiple of its MEPS-IC share. On a mixed frame (`allow_transferred_rows=True`) these proofs cover the rows with raw coverage columns; the other rows must carry the premium with a weighted positive share and mean within 0.5 to 2 times those rows'. |
| `us_esi_premiums_anchor_gate` | release tool, on the calibrated export, for both the dense and the sparse default; ACS local lane, on the staging frame and the calibrated artifact | The column absent or zero-mass is red. The anchor-universe total, recomputed at the release's weights, must be within **5%** of BEA NIPA 7.8 line 17 for the release period. Employed policyholders must carry 80% to 95% of it. On a mixed frame (`allow_transferred_rows=True`) the total adds the transferred column at the source-derived rows' employed share, and the transferred rows' column per unit of household mass must stay within 0.8 to 1.25 times the source-derived rows'. |
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
- **A mixed frame is graded by row kind, only when the caller allows it.**
  With `allow_transferred_rows=True`, which only the ACS local lane passes,
  rows with every raw coverage column get the full recomputation. Rows with
  none of them and no CPS record id (`PERIDNUM`) are the ones a transfer
  filled; the gates compare them with the first kind and cannot check them
  against MEPS-IC cells. Four things fail both gates: a row with only some of
  the columns; a CPS record without them (lost coverage codes are never graded
  as a transfer); a row with them but no record id, in a frame that has the id
  column; and a row without them in a frame that has no id column. The signal
  gate also fails when a transferred person's support clones disagree. When
  the transferred rows hold no household mass, the gates skip the comparison
  between the two kinds: there is no second half to compare. When the rows
  with the raw columns hold no household mass, or their policyholders or
  employed policyholders carry no weighted MEPS-IC share, or the employed
  policyholders' part of it rounds to zero, there is nothing to price the
  transferred rows against, and both gates fail with that refusal. When those
  rows' premium per unit of household mass rounds to zero, the per-mass
  comparison has no value and the anchor gate fails it. The anchor gate's
  private-sector cross-check scales those rows' sector total by their share
  of household mass. When that share rounds to zero, or the sector total or
  the scaled total is not finite, the cross-check is recorded without a value
  and its `unavailable` reason names which. Under the
  default, which the base build and the release tool keep, every row must
  carry every raw coverage column and the record id is not read (see "The
  gates grade the lane's frame by row kind").
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
carries the anchor verdict, both cross-checks, the base and export signal
verdicts and the anchor-versus-feed verdict, and both manifests record it.
`--allow-esi-premium-gaps` stops the gates failing a diagnostic build, for
example on a base built before this stage; the failures it waives, including
a feed whose B4923C facts differ from the anchor, are still written to that
file.

## Lineage

- Census person members: pinned in
  `education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES` (archive
  and member SHA-256), read by the #720 restoration.
- MEPS-IC cells: three AHRQ PDFs pinned by URL, byte length and SHA-256 in
  `tools/build_us_meps_ic_esi_cells.py`, extracted to
  `us_runtime/data/meps_ic_esi_premium_cells.json`, whose own SHA-256 the
  stage pins and checks on load.
- BEA and NHE files: URL, byte length and SHA-256 in the stage manifest
  (`us/source_stages.json`), and URL and SHA-256 in the module; the BEA
  values also as Chronicle feed facts, checked against the module by the
  release tool. The NHE
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
- **An ACS local release from an older donor.** The lane transfers the column
  from its donor release. A donor built before the stage has nothing to
  transfer, and the lane refuses it unless a diagnostic build waives the
  gates.
- **ACS-native predictors for the fill.** The ACS local lane fills ACS rows
  from the transfer's fixed predictors: age, sex and State, and wages,
  self-employment income, Social Security, retirement and investment income,
  household head and tenure where both sources observe them. It does not use
  ACS's own employer-coverage (`HINS1`), class-of-worker (`COW`) or
  employment-status (`ESR`) items.
- **A second verdict on the transferred rows' cells.** The gates recompute the
  MEPS-IC assignment on the rows that carry the raw coverage columns. They
  compare the transferred rows with those rows and cannot check them against
  the cells.
- **A calibration target.** The column is the employed part of the BEA fact,
  so the fact cannot be compiled as a sum target on it. The target-parity
  manifest keeps the BEA family and the two `cms_nhe` ESI families as
  `deferred` reviewed exclusions. Whether and how to calibrate the column is
  decision d1264.
- **The standalone L0/refit export.** `tools/export_us_l0_refit_h5.py`
  writes an H5 without running the signal or anchor gate. By default it
  refuses a frame whose column is absent or constant, and its generic
  input-mass parity check (default tolerance 50%) compares the column's
  weighted total with the base or reference, as for every input. Neither ESI
  gate runs there: they run only in the base build and in
  `tools/build_us_fiscal_refresh_release.py`.
- **The pre-tax employee premium.** See the section above.
