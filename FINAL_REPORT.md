# Final report: candidate-chain smoke CD-vintage provenance defect

Date: 2026-08-24

Branch: `candidate-25pct-runbook`

## Outcome

**STOP — current main has a real stacked-pool/release contract defect. There is
no honest stage-1 or stage-2a runbook argument fix.** The generated US target
contract always includes congressional districts and requires a vintage
crosswalk
(`packages/microcosm-build/src/microcosm/build/us/spec/calibration.yaml:12-20`).
The release always selects the packaged crosswalk when no override is supplied
(`tools/build_us_fiscal_refresh_release.py:1509-1516`), then requires the input
H5 to already carry matching root attrs and a positive household CD lookup
(`tools/build_us_fiscal_refresh_release.py:2565-2613,8571-8590`). The production
stacked-pool CLI exposes no crosswalk, Ledger/geography-ladder, or CD-assignment
input (`tools/build_us_multispine_pool.py:441-577`), as its exhaustive parser
test confirms
(`packages/microcosm-build/tests/test_us_multispine_pool_tool.py:4185-4247`).

The complete end-to-end investigation and exact missing wiring are recorded in
`experiments/candidate_25pct/cd_vintage_provenance_defect.md`.

## Why this is not runbook wiring

- Stage 1 supplies all six required authenticated input pairs plus the relevant
  sampling, checkpoint, and output controls
  (`experiments/candidate_25pct/run-candidate.sh:1153-1173`). The pool rejects
  preassembly operator outputs, including household CD geography
  (`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:346-353,372-406`;
  `tools/build_us_multispine_pool.py:4566-4584`), so the lookup cannot be
  smuggled through an existing source either.
- Stage 2a authenticates the correct packaged crosswalk but omits an explicit
  crosswalk flag
  (`experiments/candidate_25pct/run-candidate.sh:91-92,248-259,1189-1211`).
  Adding that flag would be a no-op because the release parser already supplies
  exactly that default (`tools/build_us_fiscal_refresh_release.py:1509-1516`;
  behavior pinned at
  `packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:1783-1822`).
- The release crosswalk translates old-vintage Ledger facts into
  current-vintage target facts; it does not assign households
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-967`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/congressional_district_vintage.py:211-298`).
  Preflight runs before Ledger load and target compilation
  (`tools/build_us_fiscal_refresh_release.py:8571-8590,8645-8658`).
- `--gate-congressional-district-targets` affects only later release-gate
  diagnostics (`tools/build_us_fiscal_refresh_release.py:6446-6589`); it cannot
  change the earlier support-provenance preflight.
- The stacked pool deliberately publishes fixed-format pandas entities and
  rejects table-format storage
  (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:966-1048`).
  The release guard recognizes the lookup only inside a compound
  `household/table` dataset
  (`tools/build_us_fiscal_refresh_release.py:2616-2661`). This physical-layout
  mismatch would remain even if a semantic CD column were added.
- The tracked production flow found that both assigns districts and stamps the
  two attrs is the separate PUF-support-base pipeline
  (`tools/build_us_puf_support_base.py:1329-1432,2756-2793`). That full producer
  is not an omitted argument or metadata-only post-step for the stacked pool.

## Existing smoke disposition

The unchanged 1% launcher built the six-input stacked pool and passed it to the
dense release as `--base-h5`
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/smoke/run-smoke.sh:18-25`).
Release preflight reports all three missing requirements: crosswalk SHA, target
vintage, and household `congressional_district_geoid`
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/smoke/release.log:45-67`).
Read-only H5 inspection confirmed fixed axis/block storage, no
`household/table`, no semantic CD column, and no vintage attrs, matching the
publisher/guard incompatibility
(`experiments/candidate_25pct/cd_vintage_provenance_defect.md:175-218`;
`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:966-1007`;
`tools/build_us_fiscal_refresh_release.py:2646-2661`). The packaged crosswalk
digest is the correctly pinned
`c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec`
(`experiments/candidate_25pct/run-candidate.sh:91-92`).

Per the requested stop condition:

- `experiments/candidate_25pct/run-candidate.sh` is unchanged;
- `--dry-run` is not extended to assert linkage that stage 1 cannot produce;
- no guard or vintage-module code is changed;
- no pool H5 or existing smoke artifact is changed;
- neither smoke stage is rerun; and
- `experiments/candidate_25pct/smoke_r2.md` is not created.

## Required current-main fix

A separate main-branch PR must:

1. add pinned crosswalk and real post-assembly household-CD assignment authority
   to the stacked-pool CLI/operator graph, because geography outputs are
   forbidden at the source boundary
   (`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:346-353,372-406`);
2. bind the authority, vintage, assignment algorithm, and seed into checkpoint
   identity, and record assignment provenance in the manifest; the current
   manifest authenticates terminal H5 bytes but has no CD-vintage assignment
   field (`tools/build_us_multispine_pool.py:3477-3568`);
3. extend atomic nullable H5 publication and verification to carry both
   required root attrs; the current writer has no attrs channel
   (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:902-1048`);
4. reconcile the release reader's `household/table` assumption with the
   producer's fixed-nullable contract
   (`tools/build_us_fiscal_refresh_release.py:2616-2661`;
   `packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:966-1048`);
   and
5. add a real stacked-pool-to-release-preflight integration test, because the
   existing guard tests mock the provenance reader
   (`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:1062-1121`).

## Verification

Five focused parser/default/guard/writer tests passed:

```text
test_parser_exposes_six_pinned_inputs_out_and_checkpoint_root
test_cd_targets_default_to_the_packaged_vintage_crosswalk
test_cd_vintage_support_provenance_requires_matching_h5_attrs
test_cd_vintage_support_provenance_rejects_missing_cd_lookup
test_nullable_writer_round_trips_fixed_tables_and_caller_artifact_kind

5 passed
```

`git diff --check` passed. The relevant runtime files are byte-identical to
local `origin/main` at `7b90bb1882b0248d751a64bf817ec127e5c42a47`. No build,
smoke rerun, push, publication, promotion, or staging action was performed.
