# Graph amendments 11–13 acceptance lane

Date: 2026-09-02
Branch: `acceptance-11-13`

## State

Acceptance work complete and committed. B6, C5, and D6 are present as strict
red charter tests and fail without their markers at the intended unimplemented
runtime boundaries. The graph suite, Ruff checks, and CI-group verification
pass. Runtime modules remain untouched.

One requested verification is externally blocked: the current burndown tool
unconditionally treats every per-file marker increase against `origin/main` as
a re-red. The three newly mandated markers therefore produce three required
`0 -> 1` increases even though coverage reports `missing 0`. The tests-only
ownership rule prevents changing that tool; the exact output is in `out.md`.

## Done

- Read `CLAUDE.md` and the charter's ownership and interface-freeze rules.
- Created an isolated worktree with
  `git worktree add ... -b acceptance-11-13 graph-amendments-11-13`.
- Recorded the required RED-test, marker, burndown, and verification contract.
- Rebased onto the source lane's follow-up `416bfab5`, which fixes inherited
  amendment serialization/pin/declaration-test fallout without changing this
  lane's tests-only diff.
- Synced the locked US/UK environment offline against the existing workspace
  venv, using a no-cache build-isolation workaround for the sandboxed uv cache.
- Added the B6 toy EXPAND and materialization-claim kernels plus
  `test_b6_entrants_are_declared`. The marked test xfails; `--runxfail` reaches
  `NodeRejectedError` because null lineage is not yet accepted or receipted.
- Added the C5 tolerance-bound producer and reporting gate plus
  `test_c5_tolerance_is_declared`. The declaration refusals are green, the
  marked test xfails, and `--runxfail` reaches the missing receipt tolerance.
- Added `test_d6_mass_is_partitioned` with a temporary int64 period source,
  partition-specific conserve refusal, and independently computed free-policy
  ledger expectations. Its green declaration refusal passes; `--runxfail`
  proves the current rejection omits the required period value.
- Updated the acceptance burndown and explainer test expectations from the
  inherited all-green state to exactly B6/C5/D6 red. The first full graph run
  exposed only those two stale bookkeeping assertions.
- Verified the full graph package: 203 passed and 3 xfailed.
- Verified Ruff lint and formatting plus CI test-group ownership.
- Verified the acceptance inventory itself is complete: 37 green, 3 red, and
  0 missing properties.
- Captured the exact unmarked B6/C5/D6 failures and implementation-facing
  receipt shapes in `out.md`.

## Next

- The dispatcher or ratchet owner must reconcile the new-charter exception
  with the per-file `origin/main` ratchet, then rerun
  `tools/graph_acceptance_burndown.py --verify --baseline origin/main`.
- The executor lane can flip each strict marker after implementing B6, C5,
  and D6 against the public behaviours fixed by these tests.
