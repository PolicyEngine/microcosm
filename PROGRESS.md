# Progress: round 12 remaining-stage input provenance

## State

Round 12 is in progress on `tail-stratum-support-652` from `8ba55275`. The
reported real 1% build reached the stacked `transferred` phase, then the QBI
derivation rejected `s_corp_income` as nonfinite for all 38,604 persons. The
current task is to trace the certified two-spine provenance of that input and
statically audit every input consumed by the remaining derive, seed, and
simulate phases before changing the declared stacked plan.

## Done

- Confirmed a clean checkout on the requested branch at `8ba55275`, 121 local
  commits ahead of the locally available `origin/main` at `d1714a7c`.
- Honored the no-network constraint: no fetch, push, GitHub, or build action
  has been performed.
- Read the repository instructions and PolicyEngine data-layer guidance.
- Established this committed Round 12 progress record before implementation.

## Next

- Identify the certified producer, universe semantics, and exact QBI consumer
  scope for `s_corp_income`.
- Enumerate a static, stage-by-stage input manifest for derive, seed, and
  simulate, and classify every input as materialized or declared by its use.
- Add failing contract coverage, implement the smallest provenance-correct
  plan/DAG change with required version bumps, then run the requested focused,
  issue-583, full-workspace, formatting, lint, and diff proofs without builds.
