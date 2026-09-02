# Graph amendments 11–13 acceptance lane

Date: 2026-09-02
Branch: `acceptance-11-13`

## State

In progress. B6 is committed-red-ready and has been proved without its marker;
C5 and D6 remain to implement. This lane owns acceptance tests only; runtime
modules remain untouched.

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

## Next

- Add and independently prove the honest RED C5 and D6 charter tests,
  committing after each coherent step.
- Run the full requested verification suite and write `out.md`.
