# Sample-size-aware income-v3 export successor — progress

Child worktree of the accepted income-v3 exporter head. Scope: add a separately
named, versioned **successor** export profile whose semantics are closed and
stable but whose cohort size is measured from verified outputs rather than
hard-coded, so the genuine 505-household run can be represented — while the
original strict 15-household profile keeps working byte-for-byte.

Local code and invented fixtures only. This lane does not export the genuine
run, does not install any candidate, does not run the model, and writes no
approval.

## State

- Worktree: `_worktrees/microcosm-us-income-sized-export-20260906`
- Branch: `us-income-sized-export-20260906`
- Base (accepted parent): `5c276925c723e17af175709b5602ee7ace32d530`
- Parent worktree `microcosm-us-income-v3-export-review-fixes-20260906`
  preserved, clean, untouched
- Started: 2026-09-06

## Done

1. Created the child worktree from the accepted parent head; copied the parent's
   `.venv` and repointed every editable `.pth` at this worktree, so tests here
   exercise this worktree's code and not the parent's.

## Next

- Establish the pre-change baseline (existing income-v3 + base exporter tests).
- Read both real run receipts, derive the measured-scope contract, build the
  successor profile.
