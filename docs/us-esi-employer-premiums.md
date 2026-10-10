# US employer-sponsored insurance premiums (#454)

Status as of 2026-10-10. Code: `microcosm.build.us_runtime.esi_premiums`
(stage `meps_esi_premiums`). Receipts: `experiments/us-esi-454/`.

## What the stage produces

Two PolicyEngine-US person inputs that no Microcosm stage produced. Their
consumers were read in the locked engine, policyengine-us 2.2.1:

| Input | Engine consumer |
|---|---|
| `employer_sponsored_insurance_premiums` | `gov.household.cbo_market_income_additions`, summed into `cbo_household_market_income`. No other variable or parameter list names it. |
| `pre_tax_health_insurance_premiums` | `gov.irs.gross_income.pre_tax_contributions`, subtracted from `employment_income` in `irs_employment_income`; and `gov.irs.gross_income.fica_pre_tax_contributions`, subtracted in `payroll_tax_gross_wages`. |

`tests/engine_contract/us/test_us_esi_premiums_engine.py` computes both in the
engine for one worker: a $2,000 pre-tax premium lowers `irs_employment_income`,
`payroll_tax_gross_wages` and adjusted gross income by exactly $2,000, and a
$10,000 employer premium raises `cbo_household_market_income` by exactly
$10,000 and moves none of the tax measures.

The first was a recorded parity gap: the retired eCPS derivation (archived
`42ed5d45`, `datasets/cps/cps.py` L197-271) read the CPS ASEC current
employment-based coverage fields, and every pinned processed ASEC input omits
them. The second never had a producer.

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
  row;
- a policyholder always has `NOW_GRP == 1`.

The stage refuses any frame that breaks these relations, carries a missing or
out-of-codebook value, or lacks a column. It never proxies (`has_esi` and
`NOW_GRP` do not identify the policyholder) and never defaults.

`PEIO1COW` (class of worker), `PHIP_VAL` (premiums paid out of pocket last
year) and `WSAL_VAL` are in the pinned inputs already. State is the household
`GESTFIPS`.

## Employer premium

**Universe.** An employed (`PEMLR` 1 or 2) current policyholder
(`NOW_OWNGRP` 1) whose current job has an employer (`PEIO1COW` 1 to 6).
Everyone else carries zero:

- dependents, because the policyholder carries the premium;
- non-employed policyholders, because retiree and COBRA coverage is not
  compensation of a current job;
- the unincorporated self-employed and workers without pay (`PEIO1COW` 7 and
  8), who have no employer to pay a share.

**Cells.** MEPS-IC average total premium and average employee contribution per
enrolled employee, by coverage tier:

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

Series III for 2025 is not yet published. The 2024 government cells are aged
to 2025 by the private-sector national ratio of the same tier and measure
(for example single premium, 9,025 / 8,486).

AHRQ suppresses nine small-firm employee-contribution cells in the 2025 State
tables (family: AK, DE, NM, SC, WV; self plus one: AK, KY, SC, SD). Each takes
the national small-firm cell times the State's all-sizes level relative to the
nation. No premium cell the stage reads is suppressed.

**Share.** Employer paid all (`NOW_HIPAID` 1): the cell premium. Some (2):
premium minus employee contribution. None (3): zero.

**Scale.** One factor scales every share so that the weighted total equals the
anchor for the build year.

## The anchor

| Series | CY2024 | Concept |
|---|---:|---|
| BEA NIPA Table 7.8 line 17, B4923C, "Private group health insurance" | **$977.0B** | Employer contributions as a supplement to wages and salaries, including government employers' contributions to privately administered plans. The same series is NIPA Table 6.11D line 32, and BEA's handbook (chapter 10, table 10.B, line 32) says its private and State and local estimates come from MEPS data "on insurance purchased by employers for employees". |
| CMS NHE Table 24, employer contribution to ESI premiums | $1,047.0B | The NHEA methodology paper (2024, p. 28) says the ESI estimates cover "active employees, continuation of health coverage (COBRA), and retirees"; note 1 of the table adds Medicare Retiree Drug Subsidy payments to the private-employer line. |

The stage scales to **BEA**, the concept that matches an employed-policyholder
column, and uses **NHE** as an upper bound.

Two things changed from what the issue recorded in July:

1. BEA revised 2024. The issue cited $1,002.9B, from the release of
   2025-09-26. The annual update published 2026-09-30 puts 2024 at $977.034B
   (and 2025 at $1,027.929B). The stage pins the September 2026 file by
   SHA-256.
2. The issue leaned toward NHE as "the broadest match to a person-level
   employer-paid concept". That held for a column that includes retirees. This
   column excludes them, and the measurement below shows NHE's extra mass is
   the non-employed policyholders.

**Measured concept check.** Applying the same cells to *every* current
policyholder, employed or not, reproduces NHE:

| Pool (income years) | All policyholders, raw | Ratio to NHE | Employed universe, raw | Non-employed or no-employer share |
|---|---:|---:|---:|---:|
| 2023-2025 (default) | $1,031.0B | 0.985 | $912.8B | 11.5% |
| 2022-2024 | $1,036.8B | 0.990 | $916.2B | 11.6% |

So scaling an employed-only column to NHE would move about $70B of retiree and
COBRA premiums onto workers. Non-employed policyholders are priced at
active-employee cells in this check, which overstates Medicare-age retirees
whose employer plans are supplements; that is why it is a concept check and
not the column.

## Pre-tax employee premium

**Eligible.** A person in the employer-premium universe with wages last year
(`WSAL_VAL` > 0), a reported premium (`PHIP_VAL` > 0) and an employee share to
pay (`NOW_HIPAID` 2 or 3).

**Amount.** The reported premium, `PHIP_VAL`, or zero.

**Probability.** The share of insurance-offering private establishments that
offer pretax employee contributions, by firm size: MEPS-IC 2025 Table I.A.2.j
over Table I.A.2.

| Firm size | Offer pretax contributions | Offer health insurance | Share used |
|---|---:|---:|---:|
| Under 50 | 15.9% | 32.1% | 49.5% |
| 50 or more | 87.8% | 94.8% | 92.6% |
| All (size unknown) | 35.5% | 49.2% | 72.2% |

The draw is a seeded blake2b uniform keyed by source identity, so the two
support clones of a person agree and a rerun is bit-reproducible.

Assumptions, each a known limit:

- Table I.A.2.j's denominator is all establishments. Its title does not say
  "that offer health insurance", unlike its neighbors I.A.2.a to I.A.2.i, so
  the stage divides by the offer rate. This assumes pretax premium
  contributions exist only where insurance is offered.
- The rates are establishment-weighted. Large establishments hold most
  enrollees and offer pretax contributions more often, so the column is a
  lower estimate.
- Government employers take the private firm-size rate for their `NOEMP`.
- `PHIP_VAL` covers every premium the person paid last year, not only the ESI
  employee share.
- Coverage is measured at interview and the premium over the prior calendar
  year.

## Measurements on the pinned inputs

`experiments/us-esi-454/run_stage_on_pinned_pool.py` pools the real pinned
inputs the way the base build does and runs the stage. These are
pre-calibration figures at pooled ASEC weights, target year 2024.

| | 2023-2025 (default) | 2022-2024 |
|---|---:|---:|
| Current ESI policyholders | 92.4M | 92.9M |
| Employed, with an employer | 79.7M | 79.8M |
| With a positive employer premium | 74.0M | 74.1M |
| Raw MEPS-IC total | $912.8B | $916.2B |
| Scale factor | 1.070 | 1.066 |
| Employer premium total | $977.0B | $977.0B |
| Mean per policyholder with a positive premium | **$13,209** | $13,193 |
| Mean per employed policyholder | $12,260 | $12,245 |
| By tier: family / self plus one / self-only | $414.7B / $208.0B / $354.3B | $415.7B / $210.1B / $351.3B |
| By employer: private / State and local / federal | $756.7B / $180.7B / $39.6B | $756.0B / $180.8B / $40.2B |
| Pre-tax premiums | $181.1B over 52.5M | $174.6B over 52.5M |
| Share of eligible workers selected | 86.5% | 86.6% |

The private-employer total ($756.7B) sits beside NHE's private line ($753.4B).
The State and local total is well under NHE's $247.0B, which includes
government retirees.

Both the signal gate and the anchor gate pass on each pool, before and after
the PUF-support clone, and the clone conserves both totals.

## Gates

| Gate | Where | Checks |
|---|---|---|
| `us_esi_premiums_signal_gate` | base build, before and after the clone; release tool, on the base and on the calibrated export | Columns present, finite, nonnegative and non-constant; support clones agree; the weighted share of people with a positive value is plausible. With the raw ASEC columns still on the frame: no employer premium outside the universe or where the employer pays none, every premium is one common multiple of its MEPS-IC share, the raw share per holder is between $6k and $20k (a broken cell table cannot hide behind the scale factor), and every pre-tax premium is the reported premium of an eligible person. |
| `us_esi_premiums_anchor_gate` | release tool, on the calibrated export, for both the dense and the sparse default | Either column absent or zero-mass is red. The employer premium total must be within **5%** of BEA for the release period and at or below NHE's employer contribution. Pre-tax premiums must be at or below NHE's employee contribution ($382.1B). |
| `us_release_input_coverage_gate` | release tool | Both inputs are `required` with no reviewed exclusion, so an export that drops or flattens either fails. |
| reform-coverage smoke | release tool, on the written file | Neutralizing the employer premium must lower `cbo_household_market_income` by at least $500B; neutralizing the pre-tax premium must raise `income_tax` by at least $5B. |

The 5% tolerance is wider than BEA's own revision of 2024 between vintages
(2.6%) and narrower than the NHE-BEA gap (7.2%), so a release that drifts to
the retiree-inclusive concept fails.

The stage pins the pre-calibration total exactly. Nothing calibrates the
column, so the anchor gate is what holds a release's total after reweighting
and sparse selection. The anchor gate's verdict ships as the `esi_premiums`
gate-evidence artifact (`esi_premiums.json`) and both manifests record it.
`--allow-esi-premium-gaps` records the gates without failing a diagnostic
build, for example on a base built before this stage.

## Lineage

- Census person members: pinned in
  `education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES` (archive
  and member SHA-256), read by the #720 restoration.
- MEPS-IC cells: four AHRQ PDFs pinned by URL, byte length and SHA-256 in
  `tools/build_us_meps_ic_esi_cells.py`, extracted to
  `us_runtime/data/meps_ic_esi_premium_cells.json`, whose own SHA-256 the
  stage pins and checks on load.
- BEA and NHE files: URL and SHA-256 in the stage manifest
  (`us/source_stages.json`) and in the module.
- The base summary records the stage's full summary twice
  (`esi_premiums_signal`, `esi_premiums_post_clone_signal`): scale factor, raw
  and scaled totals, totals by tier and employer sector, weighted
  policyholder counts, the anchor and the cell-table digest.

## What this does not do

- **The multispine pool.** `multispine_pool.py` runs its own operator chain on
  CPS-source rows that carry only part of the stacked population's weight, so
  scaling to a national total there needs its own design. Until then the
  coverage and anchor gates are red on that lineage.
- **A calibration target.** The pinned feed banks only the NHE series. The
  target-parity manifest keeps the two `cms_nhe` ESI families as `deferred`
  reviewed exclusions and records BEA's series as a source-absent family.
  Banking B4923C in the ledger would let the column be calibrated, not only
  gated.
- **The engine's medical-expense base.** In policyengine-us 2.2.1,
  `medical_expense_health_insurance_premiums` counts the whole reported
  premium (`health_insurance_premiums_without_medicare_part_b`), so a premium
  marked pre-tax here also sits in the itemized medical-expense base.
- **The wage targets.** `fiscal_targets.SOI_AMOUNT_MEASURE_VARIABLES` maps SOI
  wages and salaries to `employment_income`, while the engine taxes
  `irs_employment_income`, which is net of pre-tax contributions. This stage
  widens that existing gap by the pre-tax premium total.
