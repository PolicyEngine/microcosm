# Native survey SPM source inputs

The native SPM bridge qualifies two source inputs for income year 2024:
`is_spm_independent_minor_role` on people and
`spm_unit_spm_universe_status` on SPM units. It does not compute thresholds,
resources or poverty, certify a dataset, calibrate weights, or authorize a
release. A detached Frame or copied receipt does not carry source authority.

## Source meaning

ACS requires explicit native SPM construction and the named
`acs_spm_household_only_analysis_2024` profile. The profile retains separate
choices for admitting modeled and approved-inference membership. Its per-unit
classification uses the reviewed construction evidence, without regrouping.
Ambiguous secondary relationships make the household unresolved. Missing roles
remain missing. Group-quarters records are outside this **restricted ACS
household-only analysis**, not outside every possible SPM analysis.

ASEC requires the versioned
`ASEC_2025_INCOME_2024_SPM_POLICY`. Census specifies an unrestricted SPM
person universe in the [2025 ASEC technical documentation, Appendix D,
D-3](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar25.pdf).
That instruction supports inclusion only after authenticating the exact source
people and complete original SPM unit. It does not justify including arbitrary
rows labeled ASEC. Role and count reconciliation are additional integrity
checks; they do not supply the universe policy.

The ASEC observations were collected in 2025 and refer to 2024 income. The
bridge preserves survey-time age and relationships. It does not backdate age,
infer residence history, reuse `POV_UNIV` as SPM scope, or transfer the ACS
household-only exclusions to ASEC. Omitting the ASEC policy leaves its annual
scope unresolved. A missing required role remains unknown and prevents engine
projection for that unit.

## Retained source and final membership

The source qualifier borrows the live preparation retained by the issued
enrichment run. It validates source ownership, implementation and source bytes,
and retains nullable roles, annual scope and reasons. Its validator must run
before and after consuming the borrowed values.

The pure projection helper does not grant authority. It verifies the original
person axis, survey/source years and the receiving two-clone mapping. Every
receiving SPM unit and household must correspond to exactly one original group
and clone, with its complete person roster. Moving, mixing, splitting,
truncating or duplicating groups refuses. Existing output columns cannot be
overwritten. Projection returns new columns without changing the populations,
weights, strata or mass history.

The source nullable role is preserved separately. Only an explicitly `OUTSIDE`
person may receive a chosen Boolean storage placeholder at the final engine
boundary. Null `INCLUDED` or `UNRESOLVED` roles refuse. Known roles with
`UNRESOLVED` scope retain that status for the country's `SPM_UNIVERSE_REQUIRED`
gate. The annual status is never carried into a different model year.

The adapter honors the consuming country's explicit `spm.DATASET_SOURCE_INPUTS`
declaration, including both source inputs. Its runtime and import-free metadata
paths reject malformed, overlapping or unsupported declarations. Formula-owned
thresholds and resources remain rejected dataset columns, and the two source
inputs receive no invented defaults.

## Enrichment graph

`run_us_survey_enrichment` accepts three optional SPM arguments:
`spm_acs_profile`, `spm_asec_scope_policy` and
`spm_outside_role_placeholder`. Omitting all three keeps SPM disabled. Enabling
it requires a typed `ACSAnalysisProfile` for 2024 and an explicit Boolean
OUTSIDE placeholder. The ASEC policy may remain `None`, preserving unresolved
scope. Incomplete or untyped configuration refuses before enrichment fitting.

The host qualifies its real retained preparation and preflights the complete
receiving membership. Three ordinary nodes then follow the hours attachment:
`survey_spm.source` retains private source evidence, `survey_spm.project`
transports it at the person and SPM-unit grains, and `survey_spm.attach` adds
only the two declared source inputs to the existing population version.
Source identifiers, nullable observations and mappings remain in private
artifacts. Public parameters and the final receipt record the chosen profile,
policy, representation and artifact digests.

The host revalidates the retained source on cold and required-cache paths and
independently replays all three nodes against the complete population. Final
pure source and output checks follow the last source and artifact I/O. SPM
attachment does not make a run release eligible. Public qualification can still
inspect the issued enriched owner; projecting onto it again refuses because
the two input columns already exist.

## Verification boundary

Focused tests use invented source records and detached pure transport fixtures;
they do not constitute a population measurement. Genuine source-owner tests are
separate from pure classifier and projection tests. Optional-assembler skips or
expected failures are not evidence that native construction passed.

The graph hooks and pure fragment checks do not replace a combined final-owner
cold/cache proof. That execution is a separate integration gate. Do not
interpret these helpers or the earlier development-cohort country result as
qualification of the full native population, a production wrapper, calibrated
estimates or an app deployment.
