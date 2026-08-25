# Progress: defensive correctness audit of microcosm PR #747 (round 2)

## State

Round-2 evidence collection is complete and the report is being drafted on
the immutable live PR head
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
- Verified no terminal gate weakening, membership option, waiver, staging,
  publication/promotion side effect, or seed/identity move in the new commit.

## Next

1. Finish the broader non-build terminal-gate/source/spec test run and record
   its exact outcome.
2. Write and commit `REVIEW-747.md` with the fresh HOLD verdict, round-two
   classifications, code citations, and minimum closure conditions.
3. Update and commit this journal with the final report identity.

## Historical note

The inherited stacked-pool journal remains in the live-head parent history;
this branch-local journal supersedes it only for the PR #747 audit session.
