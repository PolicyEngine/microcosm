# Activating the direct-national ACS 2024 age counts

This is one development calibration connection: eighteen disjoint national
household age counts, produced by an ordinary population operator and fed to
the existing `demographic_calibration_node`. It is **age-only development
calibration**. It is not national validation, not congressional-district
validation, and not a release.

## What is activated, and by what authority

The accepted capture
`national-acs-2024-reference-bundle-v1/national-derived-inventory.json`
(sha256 `750b624e89557944c442600befb5557847dd75be6ce439c20daa9dfea5a0baf7`) is
**evidence, not permission**. It says so about itself: `target_activation` is
false, `calibration_target_registry` is `not_created`, and every table carries
`targets_allowed: false`. Neither that flag, nor any other descriptive field,
authorizes a target. Nothing in this slice edits the bundle.

Authority is `NATIONAL_AGE_ACTIVATION` in
`microcosm.build.us_runtime.national_age_activation`: an immutable, frozen
declaration reviewed as code, binding

- the **source digest** of the inventory,
- the **selected cells** — `S0101_C01_002` through `S0101_C01_019`, the
  publisher's disjoint AGE distribution, each pinned to its exact published
  label and to the household column that answers it,
- the **target role** `calibration`,
- the **population universe** — every resident person the publisher counts,
  group quarters included, with no household-population subsetting,
- the **target period** `2024`, and
- the **age convention** (below), by prose and by a closed token the operator
  checks against its own.

`activation_digest` covers every field, including the band bounds, labels and
column names, and travels into each produced spec's notes. A changed
declaration is a different registry version and a different node key.

`activate_national_age_targets(bundle_dir)` refuses unless the pinned bytes
reproduce that declaration, and then returns a **new** `TargetRegistry`. It
creates a registry; it does not find one.

## What is authenticated

- The inventory bytes against the declared digest. The bytes that are hashed
  are the bytes that are parsed — the file is never re-read.
- Each selected raw Census response against the digest recorded for it, and
  against its own request descriptor (keyed by `sha256(url)`), so the
  request-to-response binding is checked rather than assumed.
- The single national row's `GEO_ID` (`0100000US`), `us` (`1`) and `NAME`
  (`United States`); more than one geography row is refused.
- The publisher's own metadata for each selected cell: exact label, `group`
  `S0101`, `concept` `Age and Sex`, and `predicateType` `int` — the unit is a
  count, not a rate or a median.
- Every selected cell re-derived from those response bytes through the same
  `cd_reference.classify_acs_value` annotation-precedence reader the district
  inventory uses, then required to agree with the accepted inventory cell for
  cell. No parsing is duplicated: the reader and `strict_json` are reused.

## What is refused

Malformed, duplicated, missing and absent cells; non-count, suppressed,
not-applicable and open-interval estimates; conflicting sentinel/annotation
pairs; fractional or negative counts; the wrong period, release, country or
geography; a district-derived or non-published lineage; an unapproved selector
(the published total, a `SELECTED AGE CATEGORIES` cell, a reordered, gapped,
duplicated or partial band set); a table whose reservation status is not
`inventory_only`; an inventory that claims it is itself an activation; and any
inventory whose permission flags have been flipped.

**B19001 and B25003 remain reserved same-source holdouts.** The declaration
cannot name them, the activation refuses a bundle that has relabelled either
one away from `reserved_same_source_holdout`, and no produced spec mentions
either table. Nothing here marks them fresh or unseen.

## Uncertainty

Published margins of error are at 90 percent confidence; the documented Census
divisor `1.645` turns one into a standard error, and the resulting `se` travels
on every spec with its authority cited in the notes. **The current Adam loss
does not consume standard errors** — `DemographicCalibrationKernel` declares
`consumes_se=False`, and that is unchanged here. A cell whose margin is a
controlled-estimate sentinel carries a **null** standard error, never zero: no
published variance is not zero variance.

## The counts

`microcosm.build.us_runtime.graph_national_age_counts` registers
`us.national_age_counts@1`, an ordinary source-blind population operator. It
owns one `int64` column per band on the household table:
`people_age_0_4`, `people_age_5_9`, … `people_age_80_84`, `people_age_85_plus`.

The graph contract splits its declared read surface, and the declaration
follows that split rather than working around it. `age` is an owned cell, so it
travels in an input `Slice`. Entity ids and person-to-group memberships are
structural — the executor puts them in every context table and no node owns
them, so a `Slice` cannot name them; they are declared in the node's normative
`person_columns` and `household_columns` parameters instead, and the kernel
refuses a household view carrying anything beyond its declared id column. The
operator therefore cannot see a weight, a source channel, a spine column, or
any other household attribute. It opens no file, imports no rules engine and
calculates nothing from a model. The band schema is canonical JSON in the node
parameters, so an altered schema is a different node key.

Every included person is counted exactly once. Group-quarters people are
counted with their membership intact, because the activated universe is the
whole resident population. Rows align by id, so neither the household nor the
person axis is assumed sorted, sequential or shared. The bands are contiguous
from zero, disjoint, and closed by one open top band, so a topcoded older age
lands in `people_age_85_plus` and no age is uncovered; the counts sum to the
person cardinality, and the kernel asserts it.

Refusals are explicit and nothing is dropped silently: an unknown, fractional,
negative, non-finite or non-numeric age; a membership that names no household
row; a duplicated household id; and a household row that no person belongs to
— which the population contract already forbids, so it is a broken population
rather than an empty house.

The counts are unweighted head counts. **The graph executor owns all weights**,
and the operator returns none.

## Composition

`national_age_calibration_nodes` returns the count node and the unchanged
`demographic_calibration_node` in dependency order. The count node lives in the
base population version and owns the columns; the calibration node reweights
over that base and reads exactly those columns, so the count node is its
predecessor. A registry whose measures are not exactly the declared band
columns, in order, is refused rather than half-wired.

The published total `S0101_C01_001` is checked once for coherence — the
eighteen activated bands must partition it — and then discarded. It is
deliberately **not** a nineteenth loss term.

No US frame-context artifact is written or relabelled. A context recorded
before reweighting must not be re-stamped after it; consumers read the
executor-owned calibrated population and the diagnostics edge.

## Method, and what it does not claim

The first engineering rung uses **observed interview ages from pooled source
cohorts**, read verbatim as completed years at each person's own interview, and
calibrates that observed distribution to the 2024 published counts. No birth
date is consulted, no one is aged forward, and **no claim is made that any
modelled person was observed in 2024**. The 2024 period tag names the published
reference, not an observation date. The operator implements exactly this one
convention (`observed_interview_age_completed_years`) and refuses any other, so
the declaration and the operator cannot drift apart.

Calibration fit is not a holdout test. Neither this connection nor a successful
replay certifies national or congressional-district analysis, and nothing here
is release-eligible.

## Fixtures

The refusal surface runs against an **invented** bundle built in a temporary
directory: publisher-shaped geography ids and labels, entirely made-up counts.
The pinned tests read the accepted local capture when
`MICROCOSM_US_NATIONAL_ACS_BUNDLE` points at it, and skip otherwise.

The population fixtures are equally invented: twenty fictional people in five
fictional households, covering all eighteen bands with 0/4/5, 79/80, 84/85 and
topcoded older ages on the boundaries. The real graph counts and calibrates
them through the real Adam solver, and a fresh Python process replays all four
nodes from the content store with every kernel forbidden to run. No Census
value, registry activation or genuine population calibration is established by
any of that.

## Root review amendments, 2026-09-06

The development count operator declares a 0–120 input age envelope. Values
outside it refuse the run rather than silently entering the open 85-plus band;
the receipt reports the minimum and maximum observed ages. This envelope is an
operational guard, not validation of source age or evidence of population
coverage. The operator's age convention must agree with the activation.
Household IDs retain exact signed 64-bit values; complex or overflowing casts
are refused. Pinned reference tests require an explicitly configured
`MICROCOSM_US_NATIONAL_ACS_BUNDLE` directory.
