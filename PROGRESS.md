# Graph amendments 11–13 acceptance lane

Date: 2026-09-02
Branch: `acceptance-11-13`

## State

In progress. B6 and C5 are committed; D6 is committed-red-ready and has been
proved without its marker. Full verification and the final report remain.
This lane owns acceptance tests only; runtime modules remain untouched.

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

## Next

- Run the full requested verification suite and write `out.md`.
