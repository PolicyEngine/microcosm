# Native line integration on main — 2026-09-23

This record covers the `native-integration-20260923` lane for
[microcosm#956](https://github.com/PolicyEngine/microcosm/issues/956).
It preserves the native development stack, integrates current main, and keeps
survey poverty comparison-only. It records code integration, not an actual-data
run, artifact certification, publication approval, or a completed native release.

## Superseded review stack

The integration PR consolidates these earlier review branches. Their recorded
remote tips were each verified with `git merge-base --is-ancestor` against the
integrated lane head; this does not close or merge their PRs.

| Earlier PR | Included branch and checked tip | Scope carried into integration |
| --- | --- | --- |
| [#893](https://github.com/PolicyEngine/microcosm/pull/893) | `microcosm-us-launch-integration-20260909`, `a402dfcc9` | Original US graph-native integration |
| [#935](https://github.com/PolicyEngine/microcosm/pull/935) | `native-verify-once`, `c6e38d6f1` | Verification reuse within a native run |
| [#945](https://github.com/PolicyEngine/microcosm/pull/945) | `native-scale-transport`, `ae3955cd9` | Receipt transport and population-retention work |
| [#949](https://github.com/PolicyEngine/microcosm/pull/949) | `native-row-ceilings`, `0319b1c73` | Reviewed native row-count ceilings |
| [#950](https://github.com/PolicyEngine/microcosm/pull/950) | `native-retention-seal`, `98b698593` | Content seals and observer detachment |

The resume point was `c15661e5a`, on top of `native-release-integration-20260919`
at `47960af43` and the already integrated September 21 descendant branches.
Those completed merges were retained. Byte-transport work (`25d66f3c5`) and
verified-store-byte reuse (`78b1125f4`) were also already integrated.

## Conflict record

The earlier #950 merge (`9e9c885bc`) resolved two files: `pyproject.toml`
unions lint exceptions; `graph_atomic_survey_financial.py` combines the existing
compact profile with content seals for the default profile. Declared consumers
retain populations, the host owns their detachment, and the explicit
`_retain_every_node_population` option supports inspection. The detailed record
is in [the retention-seal note](us-native-retention-seal.md).

The resumed merge of main `d9a248d0a` had 26 conflicted paths. Their previously
resolved text was inspected and preserved; the resolver was not rerun. The
following inventory lists every conflicted path exactly once. Paths in the
tables are relative to the repository root.

| Paths | Resolution |
| --- | --- |
| `CLAUDE.md`; `docs/shared-constants.md` | Keep independent guidance from both parents. |
| `PROGRESS.md` | Preserve both historical journals and historicize the retention-seal currency claims. |
| `docs/uk-national-calibration-runbook-623.md` | Keep integration's provenance paragraph and main's driver guidance. |
| `packages/microcosm-build/src/microcosm/build/uk_runtime/__init__.py` | Preserve the lazy facade while including main's exports; adapt its export-inventory test. |
| `packages/microcosm-build/src/microcosm/build/us_runtime/__init__.py` | Keep the lazy facade and add main's SPM independence-role exports, receipt-input constant, donor entry, and stage order. The merged public export set is the exact parent union: 917 unique names. |
| `packages/microcosm-frame/src/microcosm/frame/adapters/policyengine_us.py`; `packages/microcosm-frame/tests/test_policyengine_us_metadata_index.py` | Retain integration's validated static source-input helpers and uncached declaration lookup; include main's SPM source ownership and aging changes. Keep metadata tests consistent with the declared source input. |
| `packages/microcosm-build/src/microcosm/build/us_runtime/medicaid_take_up.py` | Use main's `state_name` and `_substituted_hierarchy`, preserving the family target label; retain integration's replacement of neighboring-state metadata labels. Adapt the substitution-hierarchy test to that API. |
| `tools/build_us_fiscal_refresh_release.py`; `packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py` | Keep main's SPM composition gate and integration's formula-metadata-aware weight attachment. Move the committed-feed-pin check into shared `_compile_fiscal_release_target_registry`; preserve both test import surfaces and update the placement assertion. |
| `packages/microcosm-graph/src/microcosm/graph/executor.py` | Keep main's observer-detach behavior and documentation, alongside the integration branch's existing exact-float digest optimization. |
| `packages/microcosm-build/tests/test_uk_calibration_run.py`; `packages/microcosm-build/tests/test_us_acs_pums.py` | Union the independent test additions. |
| `tools/ci_test_groups.py`; `pyproject.toml` | Union shared-spec test registration and per-file lint exceptions. |
| `docs/evidence/spec-engine/us-f0-coverage.json`; `packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py` | Carry the merged coverage inventory and pin surface; derive final counts and hashes from the resulting code and resources. |
| `packages/microcosm-build/src/microcosm/build/us/spec/sources.yaml`; `tools/generate_us_bundle_from_constants.py`; `packages/microcosm-build/tests/test_us_bundle_core_contracts.py`; `packages/microcosm-build/tests/test_us_spec_bundle.py` | Include main's source-stage additions and corresponding source manifest, frozen-resource digest, and stage-count assertions. Derive identities from merged resources. |
| `packages/microcosm-build/src/microcosm/build/us_runtime/worker_identity.py` | Bind the approved-lock digest to the merged lockfile. |
| `packages/microcosm-build/tests/test_spec_engine_loader.py`; `packages/microcosm-build/tests/test_us_multispine_pool_tool.py`; `packages/microcosm-build/tests/test_us_spine_blindness.py` | Reconcile loader and country-spec identities and the reviewed runtime-module inventory; verify the resulting pins and counts against the merged tree. |

The subsequent merge of newer main `6442030a2` had one conflict, again in
`packages/microcosm-build/tests/test_uk_calibration_run.py`. Both independent
fixtures were retained: integration's `invented_code_pin` and main's
`_fake_cgt_projection`/`_cgt_projection` support.

`poverty-comparison-only-20260921` at `07b69be68` then merged cleanly. Its
comparison-only boundary remains explicit; poverty evidence does not supply a
population treatment or certify this integration.

## Observed identity repairs

The merged field ledger contains 42,239 distinct configuration pointers:
32,458 authored fields and 9,781 resolved bindings. Relative to the previous
42,180-field report, the validated claim changes are transfer execution +52,
producer graph -7, source stages +13, and resolved seed protocol +1.
The producer input inventory is 2,743 and the source-stage inventory is 38.
These are compiler inventories, not population coverage or release acceptance.

Two independent normal `load_bundle("us")` computations agreed on the spec
identity `217008445ad8c84088d66af45ad441b36e4f1878f4ce15f5de584fa412e8c9b6`.
The loader's separate minimal golden fixture was also re-derived from its
unchanged construction. Counts and hashes were updated from these observations;
no validation condition was removed to accommodate the merged tree.

The graph battery exposed stale accepted fingerprints in
`us_runtime/acs_native_coverage_binding.py`. Main's `59642af17` (#765) adds
optional `WKHP`/`WKL`/`FWKHP` preservation to `acs_pums.py` and annual-hours
mapping to `acs_inputs.py`. Review confirmed that missing hours remain absent,
invalid supplied codes refuse, and the existing age, relationship, and income
mappings remain intact. Only the two reviewed fingerprints were updated;
unreviewed source changes still refuse before native preparation issuance.

The later hours-fragment retry exposed a semantic integration conflict. Main's
mapper emits `weekly_hours_worked_before_lsr` when `WKHP` is supplied, while
native graph preparation reserves that column for its cross-survey hours
completion owner. The new receipt also fails the native operator-free boundary.
The mapper now accepts an explicit `include_usual_hours=False` mode: it still
validates the hours, age, past-year-work and allocation codes and retains all raw
observations, but leaves the canonical output and receipt to the completion
owner. Authenticated housing preparation and the raw ACS graph codec use that
mode; the mapper's default behavior remains unchanged. Its reviewed source
fingerprints were refreshed.
The mapper, source-hours, graph-hours and implementation-inventory files passed
243 tests after this correction. The complete ACS PUMS file passed 30 tests,
including genuine archive cases checking graph-loader raw-hour preservation and
invalid-code refusal. None of the three modified producers belongs to the F0
direct or QRF seed roster; the graph codec's static dependency contract is
unchanged. The measured portable specification identities above remain unchanged.

The native implementation inventory also required main's `microcosm.frame.scaling`
module. It is included in the exact package roster and every stage that binds
that closed package, with its static dependency contract. The frame facade,
US adapter's static-aging imports, and SPM operator-output registry import have
reviewed dependency contracts. The runtime roster and dependency refusals remain
unchanged. All ten native stage identities change because they bind the inventory
JSON; this JSON is outside the F0 seed and specification hash inputs.

The spine-blindness inventory had omitted eight integrated native modules. Their
review preserves distinct source qualification, fixture validation, and numerical
execution boundaries:

| Modules | Reviewed boundary |
| --- | --- |
| `current_survey_household_domains.py` | `_project` authenticates selected origin/clone coordinates for domain projection. |
| `current_survey_other_disability_completion.py` | The qualifier defines original ASEC observations and ACS age-15-plus development-model applicability; `_receiver` checks origin/clone consistency. Unresolved ASEC observations remain unknown. Fitting and attachment remain scanned. |
| `graph_current_survey_other_disability_completion.py`; `graph_us_other_disability_host.py` | Fixed report-column and typed artifact/population selectors; explicit provenance reads remain guarded. |
| `graph_native_puf_tail_expand.py` | `provenance_columns`, `_validated_tables`, and `native_puf_tail_expand_nodes` validate invented fixture origins and declare columns. They do not authenticate actual source data; execution and weight splitting remain scanned. |
| `native_puf_tail_matching.py` | `_frame_inputs` validates fixture consistency; ranking remains scanned. |
| `native_puf_tail.py`; `survey_social_security_beneficiaries.py` | Invented donor selection/thinning and report-lot accounting respectively; both scan clean without exceptions. |

The new guard recognizes only the listed top-level function bodies. Defaults,
decorators, other declarations, and unlisted functions remain scanned. Regression
cases reject renamed boundaries and injected provenance reads. The pool tool's
import graph was re-derived at 77 modules: its previous 73-module assertion was
already stale against the resume tree's 74, and main adds three SPM modules.

Five existing modules also required reviewed dynamic-selector entries: ASEC CSV
byte/token indexing, retained immigration state tuples and fixed lineage keys,
state-graph artifact maps, and validated SNAP state/role estimate maps. Each new
entry is covered by injected-provenance regressions; none receives a module-wide
source-provenance exemption.

The complete scan exposed unnecessary Cartesian expansion of dictionary values
while resolving dictionary iteration keys. Dependency selection now includes
ordinary keys and unpacked mappings for a root dictionary, while other expression
roots retain complete traversal. Dictionary values are still scanned separately.
A deterministic call-count regression covers the former `25**5` case; explicit
key, value, unpacking, and formatting cases verify continued provenance detection.

## Enrichment population replay direction

The first complete disability-host fixture reached the inherited population
comparison and refused `SURVEY_POPULATION_REPLAY_NONCANONICAL_NULL_BACKING`.
The replay comparator deliberately permits ContentStore normalization only on
its actual side. Review found two different comparison contexts that require
explicit ordering:

- Inherited observer populations are cached materializations of a retained
  parent. Both enrichment and disability continuation now use
  `_compare_inherited_manifest_population`, with the retained parent manifest
  as expected and the observed cached population as actual.
- Newly reconstructed expected values precede the current execution manifest.
  Disability's `_compare_manifest_population` keeps that direction for its
  final output checks, matching the existing enrichment output comparison.

The strict comparator and the separate checks of full predecessor ownership,
design weights and custody remain unchanged. Real ContentStore roundtrips for
nullable integer and boolean columns reproduced the inherited-direction refusal
before correction. The complete population-context file then passed 24 tests
across both contexts, including refusal of changed present values, masks and
noncanonical backing data. The separate replay file passed 248 tests. The exact
column from the original long fixture was unavailable after temporary-store
cleanup and is not inferred from these small reproductions. No reviewed source
pin, F0 seed hash or implementation inventory contract required adjustment;
enrichment's normal runtime source hash binds the changed implementation.

## Enrichment fixture member pins and default amount groups

The survey-enrichment module fixture refused before geography with
`DEMOGRAPHIC_SOURCE_CONTRACT`. The wrapped refusal was `STUDENT_SOURCE_SIZE`
from the demographic owner's member snapshot. The fixture pins the coverage,
restoration and demographic owners to its rewritten 2022-2024 ASEC person
members. `add_hours_source_fields` then rewrites those members again but
re-pins only the coverage and restoration owners. This did not come from main
or #991. The same refusal reproduces on the pre-merge native tree `c15661e5a`.
It also reproduces at `ac15e9fb9`, the native commit that added the hours
helper to this fixture. That commit's parent reconstructs the demographic
source successfully. After the hours helper, the fixture now re-pins the
demographic owner to the final roster, as the immigration enrichment fixture
already does. A new fixture test fails without that line; it checks the
coverage pins against the final member bytes and that the restoration and
demographic owners match them. The other owners that pin these members
(for example `asec_student_controls`) are not covered by that test.

Clearing that refusal exposed two assertions that the native line's
20 September opt-in amount routes had made stale. Those routes are workers
compensation, child support and veterans benefits. The assertions enumerated
every declared `values.GROUPS` entry, but the default call runs only
`unemployment` and `health_costs`, as `us-current-survey-amount-successor.md`
documents. A diagnostic run of the unchanged assertions found that only the
three opt-in outputs were absent from the population. They also remained
missing inputs. The node count (281) and input inventory (161) matched their
existing assertions. The tests now read the executed groups from the run
receipt and assert that default. They also require the unselected opt-in
outputs to stay absent and missing. No production module, source pin,
implementation inventory or F0 identity changed. The complete file passed
12 tests from a cold store in 1,054 seconds with an 0.87 GB maximum resident
set.

## Immigration pair producer-key control

The immigration enrichment file's
`test_genuine_pair_artifact_and_producer_controls` expected
`ARTIFACT_PRODUCER_KEY` after replacing only the retained pair artifact's
`producer_key`. The enrichment host's `context` passes each typed input through
the shared `graph_puf_detail_transfer.artifact` check before comparing
producers. That check requires `key == opaque_artifact_key(producer_key,
artifact)`. `ArtifactValue` stores `key` and `producer_key` as separate fields,
so the mutation always refused there with `DETAIL_ARTIFACT_IDENTITY`.

The merge did not change the check order, and producer detection still works.
The test, the host's `context`, the shared check, `ArtifactValue` and
`opaque_artifact_key` are byte-identical on the native parent `c15661e5a` and
the merge. The assertion arrived with native commit `d33d71bee`, which is on
neither main nor poverty-comparison-only. Its check order was already the
current one. The survey-enrichment host test has re-derived the store key for
its producer-key case since `091261c0e`.

The test now covers both refusals separately. The unmodified retained inputs
must pass first. A producer-key-only mutation must then refuse with exactly
`DETAIL_ARTIFACT_IDENTITY`. A foreign producer key whose store key is re-derived
from it passes that identity check and must refuse with exactly
`CURRENT_SURVEY_AMOUNTS_ARTIFACT_PRODUCER_KEY`. The two codes differ in form
because the two checks use different `require` helpers. The host module binds
`require = values.require`, which is `current_survey_amounts.require` and
prefixes every host refusal with `CURRENT_SURVEY_AMOUNTS_`. The shared check
binds `puf_detail_transfer.require`, which raises the bare reason. Both helpers
were already bound this way at `d33d71bee`. The sibling survey-enrichment test
matches the unanchored substring `ARTIFACT_PRODUCER_KEY`; this test anchors both
codes so that neither refusal can satisfy the other's assertion. No production
module, source pin, implementation inventory or F0 identity changed.

## Merge of main `2bce66156` (26 September)

This merge brings in the 19 main merges since `6442030a2`:

- US release: the Route A remediation stack, PR-1 to PR-5 and a follow-up
  (#1016, #1018, #1017, #1025, #1024, #1028). Also the tips return-count
  exclusion at every vintage (#1029, d179), the national + state target
  surface (#1005) and its source coverage (#1008), and the QRF tail-register
  batching (#1004). Also the capital-gains tail clone on a PUF-support base
  (#992), the stored-input contract (#1031, #1026) and the publish-now,
  flip-later tag flow (#1015).
- US inputs and grading: the PUF imputation no longer predicts income from
  itself (#1033, #982), and eCPS parity grades under the live engine names
  (#994).
- UK: bus fares from the NTS (#954), per-line release pointers and the
  2024-25 national default (#966), and income anchors at the calibration year
  (#1006).
- Shared: each packaged country spec loads once per process (#1010).

It had eight conflicted paths. The earlier resolutions recorded above were not revisited;
each file below is listed once.

| Paths | Resolution |
| --- | --- |
| `.github/workflows/test.yml` | Keep the native sharded matrix and diagnostic job. Main's only change since `6442030a2` adds `--durations=25` to the four pytest calls; the native text already carries it in each. |
| `packages/microcosm-build/src/microcosm/build/us_runtime/__init__.py` | Keep the lazy facade. Add main's three fiscal-target exports (`US_FISCAL_TARGET_ALL_VINTAGE_SUPPORT_EXCLUSIONS`, `US_FISCAL_TARGET_EXCLUSION_VINTAGE_BYPASSES`, `us_fiscal_target_exclusion_receipt`) to the `fiscal_targets` export map; `__all__` merged cleanly. The public export set is the parent union: 920 names. |
| `packages/microcosm-build/src/microcosm/build/us_runtime/reform_validation.py` | Keep main's `_released_engine_state` helper and its two callers, which collect only after releasing an object that held engine state. Keep the native `default_simulate_factory` signature with explicit dataset and microsimulation constructors and SPM selection. |
| `tools/build_us_fiscal_refresh_release.py` | Carry the native explicit-consumer seams onto main's Route A structures; details below. |
| `packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py`; `docs/evidence/spec-engine/us-f0-coverage.json` | Re-derive from the merged tree (below). |
| `packages/microcosm-build/tests/test_spec_engine_loader.py`; `packages/microcosm-build/tests/test_us_multispine_pool_tool.py` | Re-derive the golden and live spec identities from the merged tree. |

One further file had no textual conflict but pinned the facade size:
`packages/microcosm-build/tests/test_us_runtime_facade_union.py` now asserts
920 names and the union digest
`5b9df20773d016ae9cb5b8f9b3669e373ec1fe57a5b6ea2e66e4197e32a08b43`. Each
parent's `__all__` has 917 names, and their union is 920; the merged
facade's digest equals the digest of that union.

### Fiscal release tool

Main's Route A PR-4 (#1025, `f1fa9f465`) replaced the frame-based reform-validation
factory, `_batched_reform_validation_simulate_factory_from_frame`, with
`_HouseholdBatchedPostExportScorer`. The scorer reads the written H5 once,
binds its digest, scores each consumer in household batches, and records a
`post_export_scoring` block in both manifests. The native line (`0b476f984`,
`54f224f2f`) had added explicit consumer seams to the old factory: formula
metadata, a dataset constructor, a microsimulation constructor and an SPM
selection. The merge carries them onto the scorer:

- The scorer accepts `formula_metadata`, `dataset_cls` and `spm` beside its
  existing `microsimulation_cls`, `dataset_from_frame` and `load_frame`.
  Supplied metadata checks the whole written frame before any batch dataset or
  engine exists. `dataset_from_frame` and `dataset_cls` together refuse.
- An explicit SPM selection is copied once at construction. Every batch engine
  and every reform system receives its own copy, and the manifest block records
  the selection actually used.
- With no explicit seam, every call keeps main's shape exactly: the default
  loader and dataset helper see their existing arguments, each engine declares
  `US_RELEASE_SPM_SELECTION`, and a reform system is built from `reform=`
  alone. `_main` supplies no seam.
- `_score_post_export_consumer`, `_write_reform_validation` and
  `_write_demographics` forward explicit seams only to a scorer they open
  themselves. An existing scorer refuses them.

Main's Route A PR-5 (#1024, `a13324e90`) batched the target materializer's base
simulation. The native seams now reach each base batch as well as each JCT
reform batch. PR-2 (#1018, `c1b0a3204`) made a build commit mandatory with a
target-frame checkpoint path; two native checkpoint tests now supply one. The
native shared helpers remain: `_main` calls
`_compile_fiscal_release_target_registry` once, and it solves the dense and
L0 arms through `_calibrate_fiscal_support`. Exact-k keeps its ladder path. Main's exclusion receipt (Route A PR-1,
#1016, `8b228eccd`, written under the manifest receipt key by #1017) stays in
`_main`, and still replays the facts, period and crosswalk that compiler
used. The native survey handoff, the ACA frame-context fix and the
private calibration attachment bridge are unchanged.

Three native test files that addressed the removed factory now address the
scorer. They keep their assertions: explicit seams reach every batch dataset,
engine and reform system, captured selections survive caller mutation, and
supplied metadata refuses before any engine work.
`test_us_fiscal_consumer_propagation.py` adds a refusal test for overrides on
an existing scorer, and a test that omitting every seam preserves the
release's calls and manifest selection. `test_us_fiscal_formula_metadata.py`
now expects four single-household batch datasets, two base and two reform,
where the unbatched base used one. `test_us_fiscal_refresh_builder.py`'s
receipt-replay test follows the compile into the shared helper. It asserts
that `_main` has no direct compile call, passes the crosswalk, and takes back
the ledger artifact whose facts were compiled.

### Identity repairs for this merge

Every value below was observed on the merged tree; none was hand-derived.
`tools/generate_us_bundle_from_constants.py --check` passes: the committed US
bundle equals the bundle generated from the merged constants. Its identity is
`2dfa51b89e740c742437168cb07f3084a2ecd7f90ad61eb13d3b12b69f315e15`, which
replaces both parents' values in the pool tool's live-constants assertion.
Before the repair, four inventory items failed:
primary predictor tuples, producer resource semantics, seed protocol and owner
map digests, and the stacked checkpoint base identity. Their observed values:

| `EXPECTED_HASHES` key | Native | Main | Merged |
| --- | --- | --- | --- |
| `primary_tuples` | `987b501c…` | `fdf23da4…` | `fdf23da4…` (main's #1033 predictor block; the native line did not change it) |
| `late_resource_semantics` | `6c4f4ec0…` | `6abbfad7…` | `c0785687…` |
| `seed_protocol` | `b1797d9b…` | `5f6d3b52…` | `6d673c33…` |
| `seed_map` | `e04c4f86…` | `dad44808…` | `acab6b8d…` |
| `full_checkpoint` | `cfa87759…` | `b1d5c652…` | `2a1ff0f7…` |

`late_schedule`, `source_manifest`, `authority` and `graph_nodes` keep the
native values, because main left them at the base values. After the repair,
`tools/spec_engine_coverage.py` reports 42,239/42,239 configuration fields and
41/41 inventory checks. The field count equals the previous integration's.
The loader's minimal golden vector binds the seed protocol implementation
digest, so it moves with it:
`f2047cb96d0d063f40c2d5eb5afa2f3821e5667e49968268d14829bc5e09ab5a`.

## Local Git recovery and validation record

The sandbox could read the original linked-worktree Git administration but could
not write it. The lane prepared Git metadata under `.integration-resume/git`
inside its assigned workspace, retaining the original head and pending merge
state. Subsequent Git mutations use that lane-local metadata. The original
administration and other worktrees were left untouched.

Final integration commit identifiers, targeted test results, and push/PR status
will be recorded by the lane owner after validation. Earlier journal test results
are historical and are not evidence that the final merged tree passed. No
actual-data native run, benchmark, certification, upload, or publication was
performed as part of this conflict documentation.
