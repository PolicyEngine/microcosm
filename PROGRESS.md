# Progress: round 12 remaining-stage input provenance

## State

Round 12 is in progress on `tail-stratum-support-652` from `8ba55275`. The
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
exclusion.

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
  the stacked checkpoint identity and the derive-stage receipt. It contains 992
  exact consumer/input rows: 33 derive, 29 seed, and 930 simulate. The simulate
  section enumerates all 863 installed PolicyEngine input variables rather
  than using a wildcard and declares the ephemeral-default behavior for every
  present-null or absent input.
- Pinned the installed PolicyEngine-US 1.764.6 SSI dependency graph at 55 input
  leaves, 62 formula nodes, and 186 edges; pinned the full engine-input surface
  at 863 names/entities and 863 declared defaults; and pinned the complete
  manifest. Independent review found and we corrected Schedule D's
  derived-stage availability and seven present-null default paths, then
  returned `VERDICT: CLEAN`.
- Verified the manifest against smoke-r9's transferred checkpoint: all 147
  engine inputs already present are classified non-absent, and the remaining
  12 future inputs are exactly Schedule D plus 11 seed-stage additions.
- Isolated the first full-workspace chunk's only failure to Round 11's new
  generic schema-v2 checkpoint byte golden. The serializer emits identical
  `e55095d2...b44ca8` bytes across repeated writes, two filesystems, and the
  available HDF5 1.14/2.0 runtimes; the independent UK schema-v2 golden remains
  unchanged and green. Corrected only the stale generic expected digest.

## Next

- Add a static, stage-by-stage input manifest for derive, seed, and
  simulate, and classify every input as materialized or declared by its use.
- Run the requested focused, issue-583, full-workspace, formatting, lint, and
  diff proofs without builds; add the changelog and final smoke-r10 prediction.
