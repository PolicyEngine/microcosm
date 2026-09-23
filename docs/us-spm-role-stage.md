# The SPM independence role as a build-stage input leaf

Every claim here was read at `us-spm-role-stage`'s head, cut from
`spm-composition-preflight` at `0e4b20de7` (PR #948, CI green on run
35346454375). The installed engine is policyengine-us 2.2.1 /
spm-calculator 1.0.0 / policyengine-core 3.32.5. This note implements the
decision Max took on 2026-09-18 ("sure we can take your rec on spm"):
`docs/us-spm-role-for-a-fresh-base.md` Option 1, shape (b). Build P's post-hoc
enrichment lane (`tools/build_us_spm_role_enrichment.py`,
`microcosm.data.source_enrichment`) is untouched and stays pinned to Build P.

## 1. Where the rule's columns enter the pool today, and where they are dropped

`SPM_ROLE_RULE = "SPM_HEAD == 1 OR (A_FAMTYP in {1,4} AND A_FAMREL in {1,2})"`
(`us_runtime/spm_role_source.py`). The rule needs three ASEC person fields.

**They never enter.** The raw inputs the pool is built from are the three
frozen `census_cps_{2022,2023,2024}.h5` files whose paths and digests base-q3's
`summary.json` records under `base_source.sources`. Read at this head:

| raw input | person columns | `SPM_HEAD` | `A_FAMTYP` | `A_FAMREL` | `SPM_ID` | `PERIDNUM` | `SPM_HAGE`, `SPM_NUMADULTS`, `SPM_NUMKIDS`, `SPM_NUMPER` |
|---|---|---|---|---|---|---|---|
| `census_cps_2022.h5` | 137 | no | no | no | yes | yes | yes |
| `census_cps_2023.h5` | 138 | no | no | no | yes | yes | yes |
| `census_cps_2024.h5` | 162 | **no** | yes | yes | yes | yes | yes |

So there is nothing to "drop": `SPM_HEAD` is absent from every vintage, and
`A_FAMTYP`/`A_FAMREL` exist only in the 2024 file. `asec_pool.py` carries every
raw person column through `pool_asec_sources` unchanged (it adds
`source_year`, `source_household_id`, `source_person_id` = `PERIDNUM`,
`source_row_id`, remaps `PH_SEQ`/`household_id`, and makes `SPM_ID` pool-global
in `_globalize_source_unit_column`), and `assign_us_unit_structure` keeps them.
That is why the phase-2 base still carries `A_FAMTYP`/`A_FAMREL` with 66.9 %
nulls and no `SPM_HEAD` at all (`experiments/893-spm-composition-base-q3-receipt.json`).
The pool-global `SPM_ID` pattern the brief points at is exactly what
`derive_spm_role_source` relies on: it refuses to compare remapped IDs and
instead reconstructs source membership through `(source_year, PERIDNUM)`.

**Where they do exist:** the pinned complete Census ASEC person CSVs
(`pppub23.csv`, `pppub24.csv`, `pppub25.csv`, income years 2022-2024), pinned in
`education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES` and re-exposed
as `spm_role_source.ASEC_SPM_ROLE_SOURCES`, cached by
`fetch_asec_education_assistance_source` under
`~/.cache/microcosm/cps/asec_education/`. The base builder already restores raw
columns from these CSVs by exact `(source_year, PERIDNUM)` join
(`_asec_raw_source_mapping_frame`: `LKWEEKS`, `ED_VAL`, `PAW_TYP`), and the
education and public-assistance stages consume them as sidecars. The role is
the same class of repair: a measured Census field the frozen H5 inputs never
carried, restored by exact Census identity, never predicted.

## 2. The model stage, and what the new one copies from it

`us_runtime/relationship_inputs.py` is the model:

- a manifest stage (`source_stages.json` `relationship_inputs`: `read_table
  person` → `derive_relationship_inputs`) loaded by
  `us_relationship_inputs_stage_spec()`, which refuses a manifest whose outputs
  differ from the runtime-owned tuple;
- a handler `derive_us_relationship_inputs_from_manifest` that refuses the
  wrong operation kind, a missing person table, unsupported parameters, a
  missing raw column, a non-integer code, and a household without exactly one
  `P_SEQ == 1`;
- `with_us_relationship_inputs(frame, *, seed, time_period)` with an
  idempotence guard (`_relationship_surface_carries_signal`: outputs present
  and each carries more than one distinct value), `run_source_stage`, alignment
  on `person_id`, a coverage refusal, and a new `Frame`;
- `us_relationship_inputs_summary` and `us_relationship_inputs_signal_gate`
  (a `GateResult` with share bands and structural invariants);
- registration in `us_source_operation_handlers`, `US_DONORS`,
  `US_STAGE_NAMES`, `source_manifest.ALLOWED_SOURCE_OPERATION_KINDS`,
  `spec_engine/schema/sources.schema.json`, `operator_boundary`,
  `l0_refit_export.US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS`,
  `release_input_coverage`, and both build tools.

`us_runtime/spm_independence_role.py` copies that shape. Differences, each
deliberate:

- **The derivation body is a call into `derive_spm_role_source`, unchanged.**
  The handler projects the person table's identity, age and Census count
  columns (plus whichever of the optional raw fields the frame carries) and the
  `spm_unit` table into a temporary H5, hashes it, and calls the certified
  derivation with that digest as `expected_parent_sha256`. Nothing about the
  rule, the join, or the refusals is re-implemented; `spm_role_source.py` is
  byte-identical to `origin/main`. The projection is the derivation's only
  input, so the `dataset_sha256` in its provenance names the projection, and
  the stage's summary records it as `frame_projection_sha256`. Since the
  re-level onto `main` (2026-09-23) the projection is written by
  `_write_role_projection` through the shared nullable-boolean boundary
  (`microcosm.frame.put_frame_table`, pandas fixed format), registered as the
  `spm_role_derivation_projection` serializer in
  `microcosm.build.frame_serializer_registry`, because `main` forbids a bare
  `DataFrame.to_hdf` sink. The derivation still reads it with `pd.read_hdf`.
  Neither writer produces byte-identical files across runs (measured on a
  small synthetic table), so the digest binds the projection within one run;
  the receipts' `frame_projection_sha256` values were never reproducible
  bytes. `pd.read_hdf` returns tables equal to the ones written under both
  writers (same synthetic check), and the derivation sees only those tables.
- **The source is a set of pinned CSV paths, not a loaded sidecar.** The
  derivation re-verifies each CSV's size and SHA-256 itself, before and after
  reading; a DataFrame handed in would bypass that. Paths reach the handler
  through `SourceRuntimeConfig.extra["asec_spm_role_source_paths"]`, keyed by
  income year, exactly the `--asec-education-source INCOME_YEAR=PATH`
  vocabulary; a missing year is fetched and verified by
  `fetch_asec_education_assistance_source`, as the education sidecar does.
- **Two tables, not one.** The stage runs with `tables={"person": ..,
  "spm_unit": ..}` because the derivation requires that SPM membership exactly
  covers the SPM table.

### The fail-closed gates

All of these already exist inside `derive_spm_role_source` and fire as
`ValueError`; the handler re-raises derivation and filesystem failures as
`SourceRuntimeError` prefixed with the stage name, and adds its own column
check first so the message names the stage. Read from `spm_role_source.py`:

| refusal | where |
|---|---|
| a required person or SPM column is absent | `_require_columns(parent, required, ...)`; the stage checks the same tuple first |
| a person has no ASEC origin (no exact 22-digit `PERIDNUM`, or no join) | `_exact_person_keys`; `"Source join has unmatched parent persons."` |
| a native SPM unit repeats a source person, or mixes two source units | `"Native SPM unit repeats a source person."`, `"... combines distinct source units."` |
| an SPM unit has more or fewer than one `SPM_HEAD` | `_reconcile_units`: `units.heads.eq(1).all()`, on the source CSV **and** on the frame's native units |
| any SPM unit still has no classified adult after the role | `_reconcile_units`: `units.adults.ge(1).all()` — `"has an unresolved zero-adult SPM unit."` |
| derived adult/child/person counts disagree with Census's own `SPM_NUMADULTS` / `SPM_NUMKIDS` / `SPM_NUMPER` | `_reconcile_units` count reconciliation, again on both sides |
| a raw field the frame carries disagrees with the pinned CSV | `_REQUIRED_RAW_CHECKS` / `_OPTIONAL_RAW_CHECKS` equality |
| the CSV changed while being read; the projection changed while deriving | the pre/post digest checks |

The earlier note said the zero-adult invariant was "missing today". That was
wrong: `_reconcile_units` has enforced `adults >= 1` since `43171405d`. What
`minor_only_units_resolved` counts is still the number that *needed* resolving,
and the stage's gate reports it beside the number that *remain* (always zero
when the stage returns).

The stage's own gate (`us_spm_independence_role_signal_gate`) then checks the
frame it produced: derivation provenance counts and the ordered
person/age/unit/role fingerprint agree with the current frame; the column is
present, Boolean-valued, and not degenerate;
`check_spm_composition` on the frame reports zero units without a classified
adult with `role_source == "source_column"`; and two plausibility bands taken
from the three pinned vintages — the role share among all persons (measured
0.601–0.603 per vintage on the phase-2 base, 0.567 on Build P; band 0.40–0.75)
and among 15-to-17-year-olds (measured 1.49 %–1.73 % per vintage; band
0.3 %–6 %).

## 3. How clones and ACS-origin rows inherit the role

**Support clones.** The lane measured that all 907,382 persons on the phase-2
base carry a 22-digit `PERIDNUM`, including the 474,859 `puf_tax_detail`
clones. `clone_us_frame_for_puf_support` copies the person table into the two
channels, so a clone carries its source person's `source_year` and `PERIDNUM`
and joins the same CSV row. `derive_spm_role_source` merges with
`validate="many_to_one"`: a source person may repeat *across* native units (the
clone case) and is refused *within* one native unit. The receipt confirms it:
432,523 distinct source persons, **zero** with disagreeing clone roles.

Placement makes this moot for the base builder: the stage runs in
`pre_clone_enrichment`, immediately after `relationship_inputs`, on the
ASEC-only frame; cloning then copies the Boolean. It also runs correctly
post-clone (the phase-2 receipt is a post-clone frame), which is what the
release tool's `pool_frame is None` path relies on for a base built before the
stage existed. An existing role is reconciled with the pinned CSV again:
null, non-Boolean or disagreeing observations are refused, and equal roles
are preserved with renewed derivation provenance. Merely carrying a
nonconstant column never bypasses the source checks.

The metadata describes the stage input/output population at derivation time.
Cloning copies that historical metadata; its counts and binding do not attest
the expanded population until the wrapper runs again. The release path does
that before applying the role gate. The binding includes row order and pandas
value representation, so dtype conversions also require renewed derivation.
It is not a storage-independent semantic hash or a certificate for arbitrary
selected frames. The derivation's metadata contract contains JSON values;
set-valued provenance is outside that contract.

**ACS-origin rows.** These exist only in the production stacked pool
(`tools/build_us_multispine_pool.py`, `docs/us-multispine-operator-ordering.md`):
`acs_inputs.py` derives `is_household_head` for them from `RELSHIPP == 20`, and
they carry no `PERIDNUM`. The stage cannot deliver a role to them, and it must
not: `policyengine_us/spm.py` says the source declaration "does not permit
synthesizing a default value when data are absent", and the brief says never
to invent an adult where the source delivers none. So the stage **refuses** a
frame with any person lacking an ASEC origin (the unmatched-person refusal),
and this PR does **not** add it to `multispine_pool.POOL_SOURCE_OPERATOR_ORDER`.
The pool's phase-3 contract ("historical kernels run on the raw-`PERIDNUM`
CPS/ASEC availability projection and merge only their declared outputs back")
would let the stage run on the ASEC projection; what value the ACS rows then
carry is a decision (§7).

## 4. The adapter carve-out, driven by the engine's declaration

Two classification paths refuse the role today, and the release tool uses both:

1. **the live engine** (`PolicyEngineUSEngine`): `variables()`,
   `formula_owned_outputs()` and `_engine_computed_columns()` classify with
   `_is_engine_computed(variable)`, which is true for any variable carrying a
   formula — and `is_spm_independent_minor_role` carries the fallback formula
   `is_household_head | is_household_spouse`
   (`spm_calculator/policyengine_adapter.py`). `write_dataset` refuses it;
   `_engine_input_variables()` in the release tool excludes it.
2. **the import-free metadata index** (`PolicyEngineUSVariableMetadataIndex`),
   which the release tool's `_assert_no_formula_owned_columns` uses: the
   generated audit block `_GENERATED_VARIABLE_GROUPS` pins the role
   `formula_owned=True`.

The engine's own contract says the opposite about ownership:
`policyengine_us/spm.py` declares
`DATASET_SOURCE_INPUTS = frozenset({"is_spm_independent_minor_role"})` — "A
population producer must retain its observed boolean instead of treating the
fallback formula as ownership of the input." It is not in
`FORMULA_OWNED_INPUTS`, and `SPMSimulationMixin.set_input` rejects only
`REJECTED_DATASET_INPUTS`. (The earlier note said the engine asserts the two
sets disjoint; it does not — there is no assertion in `spm.py`. The carve-out
makes that check itself.)

The carve-out reads that declaration in both paths and never a name list of
microcosm's own:

- **live engine:** `_engine_dataset_source_inputs(policyengine_us)` imports
  `policyengine_us.spm`, requires `DATASET_SOURCE_INPUTS` to be a non-empty
  `frozenset` of strings disjoint from `REJECTED_DATASET_INPUTS`, and returns
  it; `variables()`, `formula_owned_outputs()` and `_engine_computed_columns()`
  treat those names as input leaves.
- **default filling:** source-deliverable names are excluded from
  `default_values()`. The disposable simulation projection refuses a present
  but incomplete role instead of replacing missing observations with `False`.
  Its audit records the role as required source data with no default; the
  stacked pool remains unqualified to supply that source input for release.
- **metadata index:** the snapshot is regenerated by
  `tools/refresh_us_generated_variable_audit.py`, which now reads the same
  declaration off the imported engine, emits `formula_owned=False` for a
  declared source input, and pins `spm.py` as a policyengine-us *activation*
  file — so an upstream change to the declaration fails the audit closed until
  it is reviewed and regenerated. The generated block is never hand-edited.

## 5. What moves, and through which generator

| artifact | why it moves | generator / procedure |
|---|---|---|
| `packages/microcosm-frame/.../adapters/policyengine_us.py` generated audit block | role becomes an input leaf; `spm.py` pinned | `tools/refresh_us_generated_variable_audit.py` |
| `test_policyengine_us_metadata_index.py` sets and counts | one name changes class | observed values from the failing assertions (no generator) |
| `us/source_stages.json` (authored) → `us/spec/sources.yaml`, `country_package.json`, `engine_abi.lock.json` | new stage; role becomes an engine input, so the remaining-stage input manifest gains a row | `tools/generate_us_bundle_from_constants.py`; the frozen `source_stages.json` digest in that tool and its two pin tests (`test_us_bundle_core_contracts.py`, `test_us_spec_bundle.py`) take the observed digest, as in `afaffae26` |
| `spec_engine/schema/sources.schema.json` | new operation kind | authored `oneOf` entry beside `derive_relationship_inputs` |
| `multispine_pool` engine contracts | +1 installed simulation input | `tools/repin_us_pool_engine_contracts.py` |
| `us/release_input_coverage_manifest.json` | role required | `tools/build_us_release_input_coverage_manifest.py` after adding it to `POST_REFERENCE_ECPS_REQUIRED_INPUTS` (module and tool) |
| seed-protocol / seed-map digests, `us-f0-coverage.json`, loader golden, `test_us_multispine_pool_tool.py` `spec_sha256`, `field_usage` counts, pointer-inventory digest | `us_runtime.source_runtime` is on `_DIRECT_KERNEL_MODULES`, and the bundle gains authored fields | `tools/spec_engine_coverage.py`; `EXPECTED_HASHES` and the count constants take observed values |
| `l0_refit_export.US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS` and its test fixture | role required non-constant in the export | authored |
| `SPM_COMPOSITION_REMEDY`, `_spm_composition_report` docstring | remedy (a) now names the stage; the "cannot be written" sentence becomes false | authored |

Not moved: `spm_role_source.py`, `source_enrichment.py`,
`build_us_spm_role_enrichment.py`, `h5_enrichment.py` (the Build P lane), the
multispine pool operator order and its registries, `asec_checkpoint.py` (no raw
column is materialized on the frame).

## 6. Proof

The original receipts below exercise `derive_spm_role_source` directly through
`experiments/spm_role_stage_proof.py`. They prove source derivation and
composition repair, but do not exercise the new stage wrapper, temporary
projection, provenance transport or weighted signal gate. Wrapper acceptance
is recorded separately; these historical receipts are preserved unchanged.

**Phase-2 base** (`experiments/893-spm-role-stage-base-q3-receipt.json`),
read-only, digest verified against its sidecar before and after:

| | before | after |
|---|---|---|
| `check_spm_composition` | FAIL | PASS |
| SPM units without a classified adult | 222 | **0** |
| units without a member aged 18 or over | 238 | 238 |
| role source | `household_structure_fallback` (`is_household_head` only) | `source_column` |

No unit re-grouped, no adult invented: person ids, SPM membership, the
`spm_unit` table and every age are byte-equal before and after; all 238
minor-only units are resolved by their own 15-to-17-year-old carrying the
source role (240 such persons; two units carry a second 15-to-17-year-old
with the role); zero unmatched persons; zero count-reconciliation
mismatches. Frame load 7.1 s, derivation 10.7 s.

**Build P agreement** (`experiments/893-spm-role-stage-buildp-agreement-receipt.json`):
the same derivation against the reviewed parent (digest
`48b9d479…`, read-only) reproduces the enrichment lane's reference evidence
**byte for byte** (SHA-256 `22b5968d…`, pin 5), 166,321 of 166,321 roles equal,
every `EXPECTED_COUNTS` value matched. On the parent itself the check fails with
28 units under the head-only fallback and passes with the role — the 28 the
lane's pin records as resolved.

### Actual wrapper acceptance, September 19

`experiments/spm_role_stage_wrapper_proof.py` independently runs
`with_us_spm_independence_role()` and its signal gate on both pinned local
populations. The final v2 receipts test committed source
`6a6d53b2fb2cad7ac7b84634decc2a41125efa92` with an empty source diff, including
the complete gate-details JSON serialization and ordered role binding.
The original derivation and first wrapper receipts remain unchanged.

| Population | Persons compared | Unresolved SPM units before → after | Wrapper seconds | Peak process RSS |
|---|---:|---:|---:|---:|
| Build P | 166,321 | 28 → 0 | 8.04 | 8.42 GiB |
| Phase-2 base-q3 | 907,382 | 222 → 0 | 9.07 | 13.28 GiB |

Both signal gates pass, with zero unmatched persons or Census count
disagreements. Every role agrees with the original derivation; Build P's
evidence bytes also equal the pinned reference. Every existing table,
column, age, membership, weight, stratum and mass log is preserved. Input
frames and pinned source files remain unchanged. The checks write aggregate
receipts only; they do not run calibration, reform validation or publication.

Receipts:
`experiments/893-spm-role-stage-wrapper-buildp-v2-receipt.json` and
`experiments/893-spm-role-stage-wrapper-base-q3-v2-receipt.json`.

Earlier wrapper receipts without `v2` predate the final JSON-serialization and
provenance-binding hardening. The projection uses a task-owned temporary H5
in a 0700 directory, deleted on exit; the receipts contain aggregate evidence.

The first review iteration passed 936 relevant tests on the pinned
PolicyEngine-US 2.2.1 / spm-calculator 1.0.0 environment. Subsequent review
found nested frozen provenance could not be written as JSON. Recursive
conversion now fixes that failure, with a real gate-to-builder checkpoint
regression. The 75 stage tests and 10 focused builder tests pass after this
fix. Newer country-model versions require separate qualification.

A second independent Fable source review closed its eight initial findings
and found no new high- or medium-severity defect. Its remaining limits were
dtype-sensitive bindings, historical pre-clone metadata and unsupported
set-valued metadata, scoped above. It did not run tests or inspect microdata;
the actual-wrapper receipts provide separate execution evidence.

All nine selected spec-engine, ABI, bundle, release-input and pool-tool test
files pass across focused reruns after correcting stale stage/input counts.
The final three-file run passes 300 tests. Bundle generation `--check` passes;
the generated coverage accounts for all 42,174 fields and 41 inventory checks.
This remains local source qualification on the pinned dependency set.

## 7. Unresolved stacked-pool qualification

**Q1 — ACS-origin rows in the production stacked pool.** The stage refuses a
frame with any person lacking an ASEC origin, so this PR does not add it to
the pool operator order. With the role now a hard-required release input, a
stacked-pool release will fail the SPM role signal gate, before calibration
or terminal input-coverage checks, until the ACS side is
decided (a reviewed exclusion is not available: the anti-rot check fails an
exclusion whose column carries signal, and Build P's enriched artifact carries
it). Options:

1. Deliver the role to ASEC-origin rows on the pool's raw-`PERIDNUM`
   projection and give ACS-origin rows the engine's own fallback
   (`is_household_head | is_household_spouse`, from their `RELSHIPP`
   headship). This changes nothing about how the engine already classifies
   them, but it stores a household-structure value under the source role's
   name — the substitution the earlier note warned against.
2. Same, but leave ACS-origin rows null and declare them in the pool's
   gap-fill schedule as a cross-origin fill. That is an imputation of the
   role; the engine reads the column as source-delivered.
3. Refuse the role on stacked pools and let those releases fail closed until
   an ACS-side source rule exists. This is what the branch does today.

**Q2 — required, or reviewed exclusion.** The coverage manifest now lists the
role as `required` with no exclusion (the #368 pattern: red on any artifact
that lacks it). If you would rather the gate stay green on artifacts built
before the stage, the alternative is a `reviewed_exclusion` — but see Q1 for
why the anti-rot check makes that inconsistent with the certified default.
