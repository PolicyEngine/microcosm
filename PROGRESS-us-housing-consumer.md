# US housing receipt/charge consumer lane

Session-handoff journal. Accurate when written; check git for current truth.

## Relation

- Parent worktree K: `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-integration-20260906`,
  branch `us-launch-integration-20260906`, frozen and clean at
  `ab10f77854b7b9e8057a07dbec40d15c10b96297`.
- `origin/main` at fetch time: `c1b8324183819cf91511c002bf724a2676524e87`. K is 124 ahead,
  0 behind; `git merge-base K origin/main == origin/main`. So this child is a strict
  descendant of both.
- This branch `microcosm-us-housing-consumer-20260906` starts at K's frozen head. K is not
  edited by this lane; its `.venv` is not created or used here.
- Environment: this worktree's own `uv sync --all-packages --locked --extra us` venv —
  policyengine-us 1.819.0, policyengine-core 3.31.0, numpy 2.4.6, pandas 3.0.3,
  CPython 3.14. The lock already pins 1.819.0; nothing was re-pinned or patched.

## Scope

Own `microcosm-us-housing-consumer-20260906` and `R/us-housing-consumer-*`. Build the
household-to-SPM housing **receipt interpretation** consumer over the already-attached
immutable `asec_housing_status` artifact, plus real invented-input probes of the pinned
engine's charge/HAP/tenant-rent behavior. Do not edit `asec_housing_status.py` or
`asec_housing_status_source.py` (a source-only authority worker may issue an
identity-changing child later).

Other lanes own CD references, housing source authority, dynamics, and the dashboard.

## State

- [x] Read AGENTS.md/CLAUDE.md, root integration decision, `R/us-reported-rent-consumer-bridge.md`,
      `R/us-housing-receipt-restoration-source-contract.md`, `docs/us-asec-housing-status.md`,
      and the frozen `asec_housing_status{,_source}.py` implementations.
- [x] Recorded K/origin relation; created dependent branch + worktree; synced pinned venv.
- [x] First exploratory engine probe against installed policyengine-us 1.819.0 confirms:
      HAP can exceed the SPM unit's own pre-subsidy charge (utility allowance in gross rent),
      `rent` floors at zero, each SPM unit in one household independently draws the
      household's full payment standard, and the capped SPM valuation differs from HAP.
- [ ] Consumer module + interpretation contract.
- [ ] Engine probes on invented households (committed tests + preserved R driver).
- [ ] Consumer tests.
- [ ] SIPP December unassisted-comparable-rent source projection requirements.
- [ ] `R/us-housing-consumer-REPORT.md` / `.json` with tests, JUnit, logs.

## Next

Implement the consumer interpretation kernel.
