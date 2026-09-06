# US CD reference reader — corrections lane journal

Corrections to the bounded, candidate-independent CD reference reader added by
`microcosm-us-cd-reference-projection-20260906`, implementing the actionable
findings of the independent Fable 5.1 review and the two root counterexamples.
No genuine projection is run here: the real 38.7 MB reference inventory is not
opened, parsed or hashed in this lane.

Branch `microcosm-us-cd-reference-corrections-20260906`, cut from the frozen
projection head `9eed2e3106ec4733e50faa70dfd8477d11f90787`, which is itself
129 commits ahead of `origin/main` `c1b8324183819cf91511c002bf724a2676524e87`
(`origin/main` is an ancestor; nothing rewritten).

## State

Started. Worktree cut, upstream relation recorded, review inputs read.

## Done

- Recorded the frozen worktree's exact relation before branching: head
  `9eed2e31`, branch `microcosm-us-cd-reference-projection-20260906`, clean
  tree, `origin/main` `c1b83241` (re-fetched, unchanged), merge-base equals
  `origin/main`, 129 ahead / 0 behind.
- Confirmed no corrections worktree, branch or report already existed from an
  automatic retry; this is the first attempt at this assignment.
- Cut `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-cd-reference-corrections-20260906`
  from that exact head. The frozen worktree is untouched.

## Next

1. Implement the seven correction groups.
2. Run the 128-test group plus the new regressions, with durable logs.
3. Write the corrections report and evidence into `R`.

## Limits held throughout

- No genuine reference projection; the real inventory is never opened.
- No protocol, adapter or pure-package file is edited.
- No candidate, source binary, microdata or held-out result read.
- No push, merge, deploy, target registration, ledger write or threshold edit.
- The frozen lane's files, reports, scripts and counterexamples are preserved
  byte-for-byte.
