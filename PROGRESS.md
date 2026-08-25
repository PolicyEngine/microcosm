# Progress: defensive correctness audit of microcosm PR #747 (round 2)

## State

Round-2 review is in progress on the immutable live PR head
`76e39f9b77256766d7608c1f03e12db2f8cef118`, on the separate local branch
`review/pr-747-r2-audit`. No push, candidate build, publication, or promotion
has occurred. The tracked worktree was verified byte-for-byte against the live
commit tree before review work began.

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

## Next

1. Recover the round-1 report from audit commit `eb50a753` and classify every
   finding as SURVIVES or RESOLVED with refreshed evidence.
2. Review the full live diff and every added receipt/register/fixture, with
   emphasis on parity honesty, fail-closed gates, doctrine, and identity pins.
3. Run focused tests and independent receipt probes without running a build.
4. Write and commit `REVIEW-747.md`, then update this journal with the final
   verdict and exact verification evidence.

## Historical note

The inherited stacked-pool journal remains in the live-head parent history;
this branch-local journal supersedes it only for the PR #747 audit session.
