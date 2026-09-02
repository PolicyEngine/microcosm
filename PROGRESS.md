# Graph amendments 11–13 acceptance lane

Date: 2026-09-02
Branch: `acceptance-11-13`

## State

In progress. The isolated worktree is based on `graph-amendments-11-13`, and
the amended interface-freeze text plus charter rows B6, C5, and D6 have been
read. This lane owns acceptance tests only; runtime modules remain untouched.

## Done

- Read `CLAUDE.md` and the charter's ownership and interface-freeze rules.
- Created an isolated worktree with
  `git worktree add ... -b acceptance-11-13 graph-amendments-11-13`.
- Recorded the required RED-test, marker, burndown, and verification contract.

## Next

- Sync the locked US/UK development environment.
- Inspect the original marker form, acceptance helpers, public runtime, and
  burndown ratchet.
- Add and independently prove the honest RED B6, C5, and D6 charter tests,
  committing after each coherent step.
- Run the full requested verification suite and write `out.md`.
