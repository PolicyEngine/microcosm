# US CD published-benchmark contract — first implementation slice

Session journal for branch `us-cd-contract-v4-first-impl`. History, not state:
check git/GitHub for current truth. Kept in its own file so the root
`PROGRESS.md` (the F1 portable worker identity journal) stays intact.

## Scope

Implement the approved canonical protocol
`us-cd-published-benchmark-v4` (SHA-256
`f7574a2a00734ce64ceb9c101ebc52469a176d37ee32b02c140d3ea2e6c568ea`, 86,626
bytes) as pure, deterministic Python over **invented fixtures only**.

Approval in force: `us-cd-contract-v4-root-implementation-approval.json`
(root APPROVE, peer APPROVE, scope `invented_fixture_implementation_only`,
`candidate_scoring: false`, `production_ledger_registration: false`).

Explicitly out of scope and not done here: scoring any actual candidate,
reading candidate/HDF/ACS/Census bundles, creating or appending a production
exposure ledger, index/registry promotion, push, deploy.

## State

Branch based on frozen `a276a3c93aed561dd08f7f209b3303fbae2e2c44` in a
dedicated worktree. Nothing pushed. No PR opened.

## Done

- [x] Verified the canonical plan digest and read it by parsed section.
- [x] Read the approval artifact, the peer review, and the independent response.
- [x] Read the reusable primitives: `holdout.py`, `quality_aggregates.py`,
      `candidate_quality.py`, `us_runtime/cd_reference.py`,
      `us_runtime/cd_reference_sources.py`, plus the two housing-universe
      adapters the protocol pins.
- [x] `microcosm.build.cd_benchmark`, 13 modules plus the packaged canonical
      document: `canonical`, `reasons`, `protocol`, `metrics`, `reference`,
      `support`, `origin`, `approval`, `ledger`, `binding`, `universe`,
      `evaluation`.
- [x] 166 invented-fixture tests across five files, all boundaries
      hand-calculable.
- [x] `docs/us-cd-published-benchmark-contract.md` and a towncrier fragment.
- [x] CI group routing: the five test files are explicit `shared-spec`, never
      `[defaulted]`; `tools/ci_test_groups.py --verify` exits 0.
- [x] Wheel build carries the packaged document at its approved digest.
- [x] Unchanged quality/holdout regression re-run green (150 tests).
- [x] Whole `shared-spec` CI group run: the only failures are five that
      reproduce identically at the frozen base
      (`test_release_target_parity.py::TestRegeneration::*` and three
      `test_spec_engine_country_bundles.py` bundle-digest parametrizations).
      They are not caused by this branch and were not re-pinned.

## Not done, deliberately

- No candidate scoring path. `evaluate_scope` refuses to read
  implementation approval as scoring approval.
- No `ProducerAuthenticator` implementation and no default `ApprovalAuthority`.
- No ledger file is ever opened, created, appended or locked.
- No national-scope claim: `/scope/national_published_reference_scope` says CD
  capture alone does not establish it.

## Next (for whoever picks this up)

- The four document gaps named in the docs' last section want a v5 append or a
  separate artifact: approval-artifact format, four ledger payload rosters,
  binding-slot reason classification, per-bin ESS floor.
- A real `ProducerAuthenticator` is the next hard piece; it must open artifacts
  and verify candidate/population binding, not echo references back.
