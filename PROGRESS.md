# Progress: round 12 remaining-stage input provenance

## State

Round 12 is in progress on `tail-stratum-support-652` from `8ba55275`. The
reported real 1% build reached the stacked `transferred` phase, then the QBI
derivation rejected `s_corp_income` as nonfinite for all 38,604 persons. The
mechanism audit is complete and implementation is beginning. The certified
processed PUF maps its combined partnership/S-corporation carrier entirely to
`partnership_income` and emits `s_corp_income` as exact zero. The historical
finalizer materialized that zero over the whole pool; the strict stacked
`preserve_nulls` path materialized it only on PUF descendants, while the
whole-pool QBI consumer retained the certified read scope. The fix will declare
and authenticate that exact whole-pool universe-zero semantic without `fillna`,
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
- Audited SSI's installed PolicyEngine-US dependency closure from the static
  source index: 55 transitive input leaves. On the smoke-r9 transferred frame,
  36 are present and complete, three SCF asset leaves are present/all-null under
  the existing explicit deferred-owner contract, and 16 are absent and
  therefore use declared engine defaults only on the disposable simulation
  projection. The complete checked-in remaining-stage manifest is being added
  so this classification cannot drift silently.

## Next

- Add a static, stage-by-stage input manifest for derive, seed, and
  simulate, and classify every input as materialized or declared by its use.
- Add failing contract coverage, implement the provenance-correct
  plan/DAG change with required version bumps, then run the requested focused,
  issue-583, full-workspace, formatting, lint, and diff proofs without builds.
