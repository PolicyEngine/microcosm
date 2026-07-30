# Progress: populace#578 increment 2

## State

- Branch: `multispine-pool-build-578`
- Worktree: `/Users/maxghenis/PolicyEngine/_worktrees/populace-578-inc2`
- Scope: code and small synthetic-fixture tests only; no dataset downloads or
  full-data builds.
- Remote `main` was verified through GitHub at merge commit
  `6c14a0a8590402d1805a24e55ca5f017f39dc281`.
- The sandbox blocked `git fetch` at DNS resolution. The local worktree is
  temporarily rooted at merged PR #581's exact head
  `956dc0a3dd5ceaffae0e2007d98dc951e773d389`, whose tree is the merge result.
  Rebase onto fetched `origin/main` remains a handoff prerequisite if network
  access is not restored in this lane.

## Done

- Read populace#578, its governing “Scope hardening” section, and the UK parity
  audit comment.
- Read merged populace#581's contract, review note, changed-file inventory, and
  merge metadata.
- Verified GitHub's current `main` tip is `6c14a0a`.
- Created the requested branch and worktree without modifying the existing
  checkout.

## Next

1. Map the pre-clone ASEC product, ACS unit-frame builder, assembly/clone/gate
   contracts, and existing multispine tool consumers.
2. Design the canonical assemble → clone → impute/derive/seed → agreement
   pipeline with sha-pinned explicit inputs and failure receipts.
3. Implement the pool builder and retain the old tool only as a thin deprecated
   shim if consumers require it.
4. Add synthetic full-path tests, changelog fragment, focused verification, and
   the external review worklog.
