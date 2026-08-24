# Candidate-chain smoke defect: stacked pool cannot carry required CD-vintage support

Date: 2026-08-24

Branch: `candidate-25pct-runbook`

Current-main authority inspected: `origin/main` at
`7b90bb1882b0248d751a64bf817ec127e5c42a47`

## Decision

**STOP: this is a current-main producer/consumer contract defect, not a
candidate-runbook argument omission.** The release's generated calibration
contract always includes congressional-district targets and declares that they
require the vintage crosswalk
(`packages/microcosm-build/src/microcosm/build/us/spec/calibration.yaml:12-20`).
The release therefore supplies the packaged crosswalk when the caller omits an
override (`tools/build_us_fiscal_refresh_release.py:1509-1516`). The production
stacked-pool parser, however, exposes no crosswalk, Ledger, geography-ladder, or
congressional-district assignment input
(`tools/build_us_multispine_pool.py:441-577`; the exhaustive parser assertion is
`packages/microcosm-build/tests/test_us_multispine_pool_tool.py:4185-4247`).

The relevant release, pool, PUF-support, H5-I/O, vintage, and calibration-spec
files produce no names under:

```text
git diff --name-only origin/main..HEAD -- \
  tools/build_us_fiscal_refresh_release.py \
  tools/build_us_multispine_pool.py \
  tools/build_us_puf_support_base.py \
  packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py \
  packages/microcosm-build/src/microcosm/build/us_runtime/congressional_district_vintage.py \
  packages/microcosm-build/src/microcosm/build/us/spec/calibration.yaml
```

The defect is therefore present in the inspected current-main implementation,
not introduced by this experiment branch. Per the task's stop condition, this
investigation does not change `run-candidate.sh`, weaken the release guard,
change the vintage module, alter the authenticated pool, extend `--dry-run`, or
rerun either smoke stage.

## Release path, end to end

1. The current stage-2a shape is supported legacy dense input mode: the release
   loads a manifest-authenticated pool when `--pool-manifest` is present and
   otherwise consumes `--base-h5`
   (`tools/build_us_fiscal_refresh_release.py:8509-8545`). A pool manifest is
   part of the all-or-none exact-k argument group and is mutually exclusive
   with `--base-h5` (`tools/build_us_fiscal_refresh_release.py:1546-1561`), so
   changing this dense smoke to `--pool-manifest` is not a provenance repair.

2. The runbook pins and authenticates the packaged crosswalk
   (`experiments/candidate_25pct/run-candidate.sh:91-92,248-259`) but does not
   explicitly pass it in the dense command
   (`experiments/candidate_25pct/run-candidate.sh:1189-1211`). That omission is
   immaterial: argument parsing always installs the same packaged crosswalk
   when no override is supplied
   (`tools/build_us_fiscal_refresh_release.py:1509-1516`), and the parser test
   pins both default and override behavior
   (`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:1783-1822`).

3. `_main` loads and hashes the selected crosswalk, then invokes
   `_assert_cd_vintage_support_matches` against the untouched base H5
   (`tools/build_us_fiscal_refresh_release.py:8571-8590`). This happens before
   the Ledger artifact is loaded and before the target registry is compiled
   (`tools/build_us_fiscal_refresh_release.py:8645-8658`), so the release cannot
   stamp or assign support before the guard runs.

4. The guard requires the H5 root attributes
   `populace_congressional_district_vintage_crosswalk_sha256` and
   `populace_congressional_district_vintage_target`, plus a household
   `congressional_district_geoid` lookup with at least one positive value
   (`tools/build_us_fiscal_refresh_release.py:2565-2613`). The canonical
   attribute names and required target vintage `119th_congress` are defined at
   `packages/microcosm-build/src/microcosm/build/us_runtime/congressional_district_vintage.py:16-32`.

5. The crosswalk passed into target compilation translates old-vintage Ledger
   facts to current-vintage target facts
   (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-967`;
   translation and lineage are implemented at
   `packages/microcosm-build/src/microcosm/build/us_runtime/congressional_district_vintage.py:211-298,604-652`).
   It does not assign congressional districts to households.

6. `--gate-congressional-district-targets` only controls how later dropped,
   skipped, and zero-support target diagnostics contribute to the release gate
   (`tools/build_us_fiscal_refresh_release.py:6446-6589`). The unconditional
   support-provenance guard has already run thousands of lines earlier
   (`tools/build_us_fiscal_refresh_release.py:8571-8590`), so this stage-2a flag
   cannot bypass or satisfy preflight.

7. Later uses of crosswalk metadata bind target-frame cache identity and report
   compilation/source coverage
   (`tools/build_us_fiscal_refresh_release.py:10356-10425,11551-11559`). None of
   those paths writes the base H5, and they execute only after the guard.

## Why the stacked producer cannot satisfy the contract

The stage-1 runbook command supplies all six required authenticated input pairs
plus its sampling, checkpoint, and output controls
(`experiments/candidate_25pct/run-candidate.sh:1153-1173`). The implementation
loads only ASEC, ACS PUMS, ACS rent, and PUF inputs
(`tools/build_us_multispine_pool.py:858-884`); no omitted runbook argument can
provide a crosswalk or household-CD assignment authority because the parser has
no such option (`tools/build_us_multispine_pool.py:441-577`).

A source-file workaround is also intentionally invalid. The operator boundary
classifies `congressional_district_geoid` as a geography-assignment output and
rejects preassembled sources that carry operator outputs
(`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:346-353,372-406`).
The stacked driver applies this check to both loaded ASEC and ACS before spine
assembly (`tools/build_us_multispine_pool.py:4566-4584`). ACS geography transfer
explicitly defers congressional districts because drawing donor geography would
sever observed state/PUMA geography
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:116-127`).

There is a PUMA-ladder assignment seam in the older optional-ACS library path:
it accepts `puma_ladder` and current-vintage expectations, then passes them into
pool assembly
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_multispine.py:69-86,154-165`).
The corresponding pool function assigns congressional districts from the PUMA
ladder (`packages/microcosm-build/src/microcosm/build/us_runtime/base_pool.py:609-688`).
The production stacked command neither accepts this authority nor calls this
path (`tools/build_us_multispine_pool.py:441-577,4566-4584`), and the optional
path contains no crosswalk-SHA H5 stamping contract.

The tracked production flow found that completes both operations is the
distinct PUF-support-base pipeline. It translates the Ledger facts, derives a
district distribution, assigns household districts, and records the assignment
inputs (`tools/build_us_puf_support_base.py:1329-1367`), then writes both vintage
root attributes (`tools/build_us_puf_support_base.py:1410-1432`; staged export
at `tools/build_us_puf_support_base.py:2756-2793`). Its parser makes assignment
dependent on Ledger facts and makes the block ladder dependent on a crosswalk
(`tools/build_us_puf_support_base.py:375-454`). Substituting that full support
base pipeline is not a stage-1 argument fix for the stacked-pool runbook.

## Independent H5-layout mismatch

The stacked publisher calls `write_nullable_us_h5` without a provenance or
root-attribute channel (`tools/build_us_multispine_pool.py:3719-3786`). The
writer signature accepts only the frame, period, artifact kind, publication ID,
and materializer version
(`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:902-951`). It
writes each entity with `preferred_format="fixed"`
(`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:966-1007`),
verifies the exact pandas round trip, and explicitly rejects table-format
entity storage
(`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:1019-1048`).
Its artifact metadata records only the fixed-nullable format, weight kind, and
optional publication/materializer identities
(`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:1093-1109`).

The release guard bypasses the format-aware pandas reader used for authenticated
stacked pools (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:775-805`).
Instead, it uses h5py and recognizes the household lookup only as a field of a
compound `household/table` dataset
(`tools/build_us_fiscal_refresh_release.py:2616-2661`). Fixed-format pandas
entities have axis/block datasets rather than `household/table`; therefore
even adding a semantic `congressional_district_geoid` column to the frame would
still fail the present physical-layout probe. The fixed-format writer contract
is pinned by
`packages/microcosm-build/tests/test_us_acs_multispine_base_builder.py:96-120`.

## Existing 1% smoke evidence

The existing launcher ran the six-input stacked pool at `--sample-fraction
0.01`, then passed its H5 to the legacy dense release through `--base-h5`
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/smoke/run-smoke.sh:18-25`).
The release traceback reports all three contract failures: missing crosswalk
SHA, missing target vintage, and missing household district lookup
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/smoke/release.log:45-67`).
The failed process used about 1.66 GiB maximum RSS
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/smoke/release.log:68-85`).

The following read-only inspection receipt is persisted here so the physical
artifact findings are reproducible rather than inferred from the writer:

```text
$ shasum -a 256 /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/smoke/pool/pool.h5
e8a84ea6bf40017c0d3108b569e4eac7e36e3fccdf4903c98b5034ab9d3a51b5  .../pool.h5

$ h5ls -r .../pool.h5 | rg '^/household(/| )|congressional_district'
/household               Group
/household/axis0         Dataset {24}
/household/axis1         Dataset {36734}
/household/block0_items  Dataset {1}
...
/household/block9_values Dataset {1/Inf}

$ h5ls .../pool.h5/household/table
table      **NOT FOUND**

$ h5dump -d /household/axis0 .../pool.h5 | rg '^   \([0-9]+\):'
(0): "household_id", "state_fips", "H_TENURE", "SERIALNO", "ST", "PUMA",
(6): "puma_geoid", "puma", "NP", "ADJHSG", "TEN", "RNTP", "GRNTP",
(13): "TAXAMT", "TYPEHUGQ", "tenure_type", "acs_monthly_contract_rent",
(17): "acs_monthly_gross_rent", "acs_annual_property_tax",
(19): "household_spine_source_id", "household_source_id",
(21): "household_support_channel", "household_support_clone_index",
(23): "household_weight"

$ h5dump -A -g / .../pool.h5 | sed -n '1,38p' | rg 'ATTRIBUTE'
ATTRIBUTE "CLASS" {
ATTRIBUTE "PYTABLES_FORMAT_VERSION" {
ATTRIBUTE "TITLE" {
ATTRIBUTE "VERSION" {
```

This inspection of the unchanged H5 found 36,734 household
rows and fixed-format `household/axis*` plus `household/block*` datasets, with
no `household/table`. Its 24 household columns include `puma_geoid` but not
`congressional_district_geoid`; its root carries only generic PyTables
attributes. Those observations are exactly predicted by the stacked fixed
writer (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:966-1007`)
and are incompatible with the release's raw table probe
(`tools/build_us_fiscal_refresh_release.py:2646-2661`). The unchanged pool H5
SHA-256 is
`e8a84ea6bf40017c0d3108b569e4eac7e36e3fccdf4903c98b5034ab9d3a51b5`.

The packaged crosswalk independently hashes to
`c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec`,
matching the runbook pin
(`experiments/candidate_25pct/run-candidate.sh:91-92`) and the release's expected
digest in the traceback. The mismatch is missing producer linkage, not a wrong
crosswalk artifact.

## History and test gap

- Commit `5a2b094b` introduced the vintage guard together with the PUF-support
  producer that assigns and stamps support
  (`tools/build_us_fiscal_refresh_release.py:2565-2613`;
  `tools/build_us_puf_support_base.py:1329-1432`).
- Commit `136ffc91` later introduced the assemble-first stacked-pool builder;
  its current parser and writer still have no corresponding geography/vintage
  input or attr channel
  (`tools/build_us_multispine_pool.py:441-577,3719-3786`).
- The #733 merge `2aa96795` is UK FRS work and the #735 merge `b4dfa0e7` is
  UK/generic Ledger-target work; neither commit's file list adds US stacked-pool
  CD wiring. The relevant current US surfaces remain those cited above.
- Commit `b7922b08` in the #741 merge `c31e1525` removed the opt-in target
  surface and made the packaged crosswalk fallback unconditional; that behavior
  is now explicit in code and tests
  (`tools/build_us_fiscal_refresh_release.py:1509-1516`;
  `packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:1783-1822`).
- Guard unit tests monkeypatch the provenance reader rather than passing a real
  stacked H5 through it
  (`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:1062-1121`).
  The tracked stacked-pool parser test simultaneously asserts the absence of
  any CD/crosswalk input
  (`packages/microcosm-build/tests/test_us_multispine_pool_tool.py:4185-4247`),
  leaving the producer/consumer integration gap uncovered.

## Exact missing current-main wiring

A separate current-main fix must implement all of the following as one
authenticated contract:

1. Add explicit, pinned household-CD authority to the production stacked-pool
   CLI: a current-vintage crosswalk plus a supported geography assignment input
   such as the Ledger distribution or PUMA ladder. The producer must run after
   source assembly because preassembly geography outputs are forbidden
   (`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:346-353,372-406`;
   `tools/build_us_multispine_pool.py:4566-4584`).
2. Bind the new authority, vintage, assignment algorithm, and seed into
   checkpoint identity, and record assignment provenance in the terminal
   manifest. The current manifest already authenticates the terminal H5 bytes
   but has no CD-vintage assignment field
   (`tools/build_us_multispine_pool.py:3477-3568`).
3. Extend nullable H5 publication to write and round-trip the crosswalk SHA and
   target-vintage root attributes atomically. The current signature and
   verifier provide no attrs channel
   (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:902-1048`).
4. Reconcile release preflight with the stacked artifact's deliberate
   fixed-nullable representation, preferably through the shared format-aware
   reader, or deliberately version/change the storage contract. The present
   reader assumes `household/table`
   (`tools/build_us_fiscal_refresh_release.py:2616-2661`) while the publisher
   forbids table-format entities
   (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:966-1048`).
5. Add a real tiny stacked-pool-to-release-preflight integration test covering
   both attributes and positive household district support. The current guard
   tests use mocked dictionaries
   (`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:1062-1121`).

Metadata-only stamping is not an acceptable subset of this work: the guard also
requires positive household lookup values
(`tools/build_us_fiscal_refresh_release.py:2604-2608`), and current source
boundaries prevent those values from being smuggled in
(`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:346-353,372-406`).

## Verification and disposition

Focused, low-memory contract tests passed:

```text
packages/microcosm-build/tests/test_us_multispine_pool_tool.py::test_parser_exposes_six_pinned_inputs_out_and_checkpoint_root
packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py::test_cd_targets_default_to_the_packaged_vintage_crosswalk
packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py::test_cd_vintage_support_provenance_requires_matching_h5_attrs
packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py::test_cd_vintage_support_provenance_rejects_missing_cd_lookup
packages/microcosm-build/tests/test_us_acs_multispine_base_builder.py::test_nullable_writer_round_trips_fixed_tables_and_caller_artifact_kind

5 passed
```

No build or smoke rerun was performed. `experiments/candidate_25pct/run-candidate.sh`
and all existing external smoke artifacts remain unchanged. Because supported
linkage does not exist, extending `--dry-run` would only claim a contract that
stage 1 cannot produce, and `experiments/candidate_25pct/smoke_r2.md` is
intentionally not created.
