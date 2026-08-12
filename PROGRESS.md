# Progress: round 12 remaining-stage input provenance

## State

Round 12 is complete on `tail-stratum-support-652` from `8ba55275`. The
reported real 1% build reached the stacked `transferred` phase, then the QBI
derivation rejected `s_corp_income` as nonfinite for all 38,604 persons. The
mechanism audit is complete and the provenance fix is implemented. The certified
processed PUF maps its combined partnership/S-corporation carrier entirely to
`partnership_income` and emits `s_corp_income` as exact zero. The historical
finalizer materialized that zero over the whole pool; the strict stacked
`preserve_nulls` path materialized it only on PUF descendants, while the
whole-pool QBI consumer retained the certified read scope. The fix declares
and authenticates that exact whole-pool universe-zero semantic without `fillna`,
while retaining QBI's exact nonfinite check and the deliberate transfer-plan
exclusion. Focused, issue #583, full-workspace, lint, format, diff, and
independent-review proofs are green. No build was run; smoke-r10 remains the
external certification step.

## Done

- Confirmed a clean checkout on the requested branch at `8ba55275`, 121 local
  commits ahead of the locally available `origin/main` at `d1714a7c`.
- Honored the no-network constraint: no fetch, push, GitHub, or build action
  has been performed.
- Read the repository instructions and PolicyEngine data-layer guidance.
- Established this committed Round 12 progress record before implementation.
- Reproduced the decisive artifact facts from the completed smoke-r9
  `transferred.checkpoint.h5`: 80,395 person rows; `s_corp_income` has exactly
  38,604 nulls and 41,791 exact zeros, with every one of the 38,604 native
  role-0 rows null and all PUF descendant rows zero.
- Traced the PUF donor construction: when the certified processed artifact
  exposes only `partnership_s_corp_income`, `partnership_income` receives the
  combined value and `s_corp_income` receives an exact zero array. The smoke-r9
  primary-QRF target bank likewise contains 23,179 exact zero draws.
- Confirmed `s_corp_income` is deliberately excluded from the ACS transfer
  family until the base disaggregates the combined carrier. Treating the
  structural zero as a new stochastic transfer target would misstate that
  provenance.
- Located both whole-pool QBI reads: reconciliation and its signal summary use
  `_optional_numeric`, which delegates a present column to the unchanged exact
  all-row finiteness check. That is why the 38,604 declared absences fail at the
  first post-transfer derive operation.
- Chosen the certified-semantics fix: a named, fail-closed stacked
  primary-PUF universe rule will require exact-zero donor and PUF-descendant
  values, require all non-owned cells to remain absent before the operation,
  and then assign an explicit whole-pool zero array with a bound receipt. The
  late registry will advertise whole-pool coverage for this one output. This is
  a declared deterministic materialization, not missing-value imputation.
- Implemented that producer after primary QRF and capital-gains-tail
  convergence. It rejects a missing, nonfinite, or nonzero donor; any
  pre-materialized native cell; and any nonfinite or nonzero clone-1/clone-2
  cell before explicitly assigning zeros to native rows. Its receipt binds the
  rule, per-role counts, and donor/person value digests.
- Advanced the late-producer registry to schema 16, stacked authority to 10,
  primary execution-resource schema to 4, outer stacked materializer to 11,
  and shared pool checkpoint envelope to 7. The callback receipt and resource
  binding carry the same named whole-pool output-universe doctrine, so older
  checkpoints fail closed.
- Added focused producer/DAG/version tests, including the exact all-null QBI
  regression, and ran the consolidated producer selection: 18 tests passed.
- Audited SSI's installed PolicyEngine-US dependency closure from the static
  source index: 55 transitive input leaves. On the smoke-r9 transferred frame,
  33 are present and complete, three SCF asset leaves are present/all-null under
  the existing explicit deferred-owner contract, and 19 are absent. The seed
  stage materializes `takes_up_ssi_if_eligible` at its disclosed engine default,
  leaving 34 complete, three explicitly deferred, and 18 absent leaves that use
  declared engine defaults only on the disposable simulation projection. The
  complete checked-in remaining-stage manifest now prevents this
  classification from drifting silently.
- Added the complete remaining-stage manifest and bound its content receipt to
  the stacked checkpoint identity and the derive-stage receipt. It contains 993
  exact consumer/input rows: 34 derive, 29 seed, and 930 simulate. The simulate
  section enumerates all 863 installed PolicyEngine input variables rather
  than using a wildcard and declares the ephemeral-default behavior for every
  present-null or absent input.

  | Stage | Consumer | Inputs | Static provision point |
  | --- | --- | ---: | --- |
  | derive | `prepare_stacked_tail_derivation` | 2 | assembled provenance or declared optional derived leaf |
  | derive | `_complete_schedule_d_input` | 5 | assembled structure or transferred parents |
  | derive | `with_us_qbi_input_reconciliation` | 27 | assembled provenance/native source or transferred declared producer |
  | seed | `seed_multispine_pool_inputs` | 13 | transferred input, administrative seed, or disclosed engine default |
  | seed | `with_us_take_up_inputs` | 16 | assembled identity, membership, weight, and age inputs |
  | simulate | `PolicyEngineUSEngine.materialize` | 12 | assembled entity graph and household weight |
  | simulate | `ssi_static_dependency_closure` | 55 | 33 complete, 3 declared deferred, 19 declared absent at transfer; 34/3/18 after seed |
  | simulate | `_simulation_projection` | 863 | 123 materialized, 13 seeded, 3 deferred, 5 native, 10 structural, 4 preserved, 1 derived, 704 declared absent |
  | **Total** |  | **993** | every read materialized by `available_by` or paired with an explicit fallback |

- Pinned the installed PolicyEngine-US 1.764.6 SSI dependency graph at 55 input
  leaves, 62 formula nodes, and 186 edges; pinned the full engine-input surface
  at 863 names/entities and 863 declared defaults; and pinned the complete
  manifest at
  `8247a93e5f8f63d3ae71c1de681c29524d4bb8f07e3c6a50dcaf431b1377020f`
  with receipt
  `54b7196a6cf7d1ae18a6b149833fe5ecf5d998b4b14f8388766341af153ff3df`.
  Independent review found and we corrected Schedule D's
  derived-stage availability and seven present-null default paths, then
  returned `VERDICT: CLEAN`.
- Verified the manifest against smoke-r9's transferred checkpoint: all 147
  engine inputs already present are classified non-absent, and the remaining
  12 future inputs are exactly Schedule D plus 11 seed-stage additions.
- Resolved the first full-workspace chunk's only failure as a test-environment
  mismatch, not a serializer change. The borrowed Populace environment lacked
  the repository-pinned PyArrow dependency and emitted `e55095d2...b44ca8`;
  restoring locked PyArrow 25.0.0 reproduces the checked-in
  `7671ab32...d930` bytes exactly. Restored the original generic schema-v2
  golden; the independent UK golden remains unchanged.
- Re-ran the complete focused Round 12 surface against a stable tree: exactly
  638 passed, with zero skips, failures, or errors. Re-ran issue #583's required
  blindness proof separately: exactly 495 passed, with zero skips, failures, or
  errors. The final locked-environment rerun of the generic and UK schema-v2
  checkpoint suites passed 33 tests with zero skips, failures, or errors.
- Independent review found one transitive manifest omission: the ACS earnings-
  universe owner reads person support channel while resolving the QBI scope.
  Added an owner-level structured input declaration, registered that physical
  input without teaching the pool operator source-channel semantics, and
  restored the exhaustive manifest to 993 rows. No other actionable finding
  remained. The nine manifest/derive regressions and issue #583's exact 495
  tests pass on the corrected tree.
- Added the Round 12 changelog fragment describing both the certified whole-
  pool S-corporation zero universe and the exhaustive remaining-stage manifest.
- Re-ran the final focused eight-file surface on the committed tree: 638 passed,
  zero skipped, failed, or errored. Re-ran issue #583 separately and asserted
  exactly 495 passed, zero skipped, failed, or errored.
- Partitioned all 228 non-#583 test files into eight sorted, disjoint chunks and
  asserted the partition cardinality. The chunks reported, respectively:
  `743/743/0`, `637/616/21`, `782/777/5`, `840/839/1`, `994/992/2`,
  `814/813/1`, `766/738/28`, and `82/74/8` tests/passed/skipped, with no
  failures or errors. Including #583, the exact 229-file workspace total is
  6,153 tests: 6,087 passed, 66 skipped, zero failed, and zero errored.
- Ran repository-wide `ruff check .`, changed-Python-file
  `ruff format --check`, `git diff --check 8ba55275..HEAD`, and working-tree
  `git diff --check`; all passed. The final independent review returned
  `VERDICT: CLEAN` with no actionable correctness, regression, or coverage
  finding.
- Recomputed the smoke-r10 identities from the exact smoke-r9 pins and stack
  receipt: configured namespace
  `2e45c4d60f66b4321bc00ffa22816470bf162c59fd91956514832f97e066ed3c`,
  base identity
  `5fa474987eb0c9f3dc461cb0e3656678ac45dd449ef1b7d683f8311c092d39d0`,
  assembled identity
  `f584881dc59088efc7b9372d154a97eb7509fa7bd4070add07b55e9855586d25`,
  transferred identity
  `f7107c4591df4ec3e4250f32923251ac418f00c2674f6fd97db13ba75a602a8b`,
  and simulated identity
  `50e4b6885bee8f05aca3f94800a78807c82ca0294d9e22d683813ed75c6e06ba`.
  The stacked authority is
  `f0b676f6508dbf6bb2b787c42e6b85331bacc57c6649ac7ad15fdaa5884a1b2d`.
  The new configured namespace cannot discover smoke-r9's
  `99376eea69594de6c88e2f68f76e35e6590a3f1cdc2849953257f0de3a7d2f46`
  subtree, so smoke-r10 must rebuild all 65 primary-QRF target files and 117
  physical ACS-transfer bank files (118 logical outputs; the immigration pair
  shares one file). The late schedule remains 38 producers, 16 source
  producers, 19 transfer groups, 70 targets, 71 edges, and six waves, with
  schedule SHA-256
  `b1d00afea69b2009d862ca73fff1b63ce56628a8a0790be49918e4bbbecc9fc5`.
- Predicted the new whole-pool S-corporation receipt on smoke-r10 exactly:
  23,179 donor rows verified; 38,604 native rows materialized; 41,791 produced
  rows verified; 80,395 person rows; clone-role counts 0/1/2 of
  38,604/38,604/3,187; and zero post-materialization nonfinite or nonzero rows.
  The attempt should pass `transferred`, then reach `derived`, `seeded`, and
  `simulated` without the reported QBI exception. Later terminal-gate outcomes
  remain certification results, not static predictions.

## Next

- Run the external real 1% smoke-r10 build and compare its identities, rebuild
  counts, S-corporation receipt, and phase sequence with the prediction above.
