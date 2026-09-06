# December SIPP housing predictor projection — root verification lane

## State

Open on 2026-09-06 on `us-sipp-housing-predictor-root-20260906`, branched from
`f20e413a130fc7bba2d6e91b3af88f3af48b26a7` on
`microcosm-us-sipp-housing-predictor-projection-20260906`, which is itself
seventeen commits on top of the accepted housing projection
`5534a0fc155dfdfe3b399725ed44fbb9d42f859e`.

That predictor branch was **not written by this session**. A separate local
session built it between 07:09 and 08:37 today and exited; its worktree
`_worktrees/microcosm-us-sipp-housing-predictor-projection-20260906` is clean at
`f20e413a` and is left untouched. This lane inherits that work rather than
rebuilding it, and its job is to verify every claim in it against the primary
documents and the code, correct what is wrong, and carry the increment to a
head whose report is true of that head.

Nothing is pushed, merged, deployed or published. No agent is spawned. No SIPP,
ASEC or ACS byte is read. No fit, draw, price bridge, donor mask, model input
binding or PolicyEngine calculation is performed.

## Lineage

`origin/main` fetched at lane open, `c1b8324183819cf91511c002bf724a2676524e87`.
`git rev-list --count 5534a0fc..origin/main` is 0 and
`git merge-base 5534a0fc origin/main` is `c1b83241`, so the accepted base
already contains the whole of maintained `origin/main` and no merge is needed.
Local `main` in the primary clone is far behind `origin/main` and is not the
lineage reference.

## Done

- Established that the target branch already existed and its author had exited,
  before making any edit. `lsof` on every live `claude` process showed none
  holding that worktree, and the two transcripts naming the branch belong to
  sessions absent from the process table.
- Opened this journal. The repository convention is one `PROGRESS-<lane>.md`
  per lane and the root `PROGRESS.md` is an unrelated F1 journal that a previous
  lane had already overwritten once; it is not touched here.

## Next

- Verify the inherited dictionary locators directly against the cached PDF.
- Strip the thirty-eight formatter-only files that commit `7b5589f0` swept in.
- Verify the frozen sixteen-field producer is byte-identical to the base.
- Re-run the suite and the replay contracts at this head.
