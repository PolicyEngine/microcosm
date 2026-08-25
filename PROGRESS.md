# Progress: defensive correctness audit of microcosm PR #747 (round 2)

## State

Round-2 review is complete with a **HOLD** verdict on the immutable live PR head
`76e39f9b77256766d7608c1f03e12db2f8cef118`, on the separate local branch
`review/pr-747-r2-audit`. No push, candidate build, publication, or promotion
has occurred. The tracked worktree was verified byte-for-byte against the live
commit tree before review work began.

GitHub reports that the PR was merged while this audit was in progress, at
2026-08-25 11:45:37 UTC as merge commit
`2807e99cc11d1a8c69a0886992ae21a533719958`. The report will therefore frame
the blocking verdict as immediate revert/follow-up evidence rather than a
remaining pre-merge check.

The required `uv sync --all-packages --extra us` resolved the lock but could
not download `ndindex==1.10.1` because the managed sandbox cannot resolve
PyPI. An offline retry could not use the host uv cache because that cache is
outside the writable sandbox. Targeted verification will use the existing
environment where possible, and this limitation will be recorded with any
affected test evidence.

## Done

- Read `CLAUDE.md` and the GitNexus PR-review skill instructions.
- Confirmed the fetched/supplied live commit object is exactly
  `76e39f9b77256766d7608c1f03e12db2f8cef118`.
- Verified the index and worktree exactly match live tree
  `ca00954e78a752a2a4ba492fad6e08120add572a`.
- Preserved the supplied `PR747-R2-COMMIT.txt` and `PR747-R2-LIVE.diff`
  unchanged as untracked evidence inputs.
- Created a separate local audit branch so no review commit lands on the PR
  branch.
- Recovered the round-1 report from `eb50a753` and rechecked every finding on
  the live tree: all five survive; `76e39f9b` resolves only the entity-table
  subpart of finding 2.
- Read the full 52-file live diff and every added receipt, register, fixture,
  resource, gate pin, and generated manifest relevant to acceptance.
- Reproduced the age-tail/NHS E6 failure with production functions: of 400
  top-coded people, 188 moved to age 85+, 67 to age 90+, and all six stored NHS
  columns disagreed with E6 recomputation.
- Reproduced strict-mode count, signed-direction, configurable-band,
  weighted-key, identity-binding, candidate-only entity, and root-attribute
  fail-open cases.
- Ran the six focused age/NHS/parity/register/payload test files: 115 tests,
  exit 0, using the compatible sibling lock environment with this worktree's
  package sources first on `PYTHONPATH`.
- Completed the four supplemental UK parity-reference, terminal-gate,
  weighted-integrity, and source-stage files: 104 passed, 1 skipped. An
  unrelated US country-package test stalled in the fallback environment and
  was interrupted.
- Verified no terminal gate weakening, membership option, waiver, staging,
  publication/promotion side effect, or seed/identity move in the new commit.
- Wrote the final code-cited report to `REVIEW-747.md`; its final report head
  is `f322ea40` (initial report commit `4ba61e90`).

## Next

1. Do not rely on or promote the already-merged UK spine swap.
2. Fix the age-tail/NHS E6 inconsistency and strict signed-parity enforcement.
3. Rebuild the exact 25-stage candidate and commit identity-bound acceptance
   receipts before clearing the HOLD.

## Historical note

The inherited stacked-pool journal remains in the live-head parent history;
this branch-local journal supersedes it only for the PR #747 audit session.
