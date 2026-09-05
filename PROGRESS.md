# US CD published-benchmark contract — first implementation slice

Session journal for branch `us-cd-contract-v4-first-impl`. History, not state:
check git/GitHub for current truth.

## Scope

Implement the approved canonical protocol
`us-cd-published-benchmark-v4` (SHA-256
`f7574a2a00734ce64ceb9c101ebc52469a176d37ee32b02c140d3ea2e6c568ea`) as pure,
deterministic Python over **invented fixtures only**.

Approval in force: `us-cd-contract-v4-root-implementation-approval.json`
(root APPROVE, peer APPROVE, scope `invented_fixture_implementation_only`,
`candidate_scoring: false`, `production_ledger_registration: false`).

Explicitly out of scope and not done here: scoring any actual candidate,
reading candidate/HDF/ACS/Census bundles, creating or appending a production
exposure ledger, index/registry promotion, push, deploy.

## State

Branch based on frozen `a276a3c93aed561dd08f7f209b3303fbae2e2c44` in a
dedicated worktree. Nothing pushed.

## Done

- [x] Verified the canonical plan digest and read it by parsed section.
- [x] Read the approval artifact, the peer review, and the independent response.
- [x] Read the reusable primitives: `holdout.py`, `quality_aggregates.py`,
      `candidate_quality.py`, `us_runtime/cd_reference.py`,
      `us_runtime/cd_reference_sources.py`, plus the two housing-universe
      adapters the protocol pins.

## Next

- [] `microcosm.build.cd_benchmark` package: protocol, reasons, reference,
      metrics, support, origin, ledger, approval, evaluation.
- [ ] Invented-fixture tests with hand-calculable boundaries.
- [ ] Docs + changelog fragment; CI group routing.
- [ ] Author report to `R/us-cd-contract-v4-first-implementation-author.{md,json}`.
