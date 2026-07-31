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
- Audited the legacy ACS multispine tool's consumers. The local-release builder
  imports its H5 helpers and the published build recipe names its CLI, so the
  legacy path will remain only as a deprecated compatibility shim.
- Traced the production stage contracts from the ASEC pre-clone checkpoint and
  byte-pinned ACS PUMS acquisition through assembly, PUF cloning and transfer,
  derivation, seeding, SSI simulation, and the fixed agreement gate.
- Identified two correctness fixes required by the new ordering:
  `transfer_acs_inputs` must fill missing cells without overwriting measured
  values, and every post-assembly `Frame` reconstruction must preserve the
  #581 metadata receipt.
- Ran the focused #581 assembly, agreement, clone-routing, and AST-guard tests
  against the starting tree: 44 passed.

## Next

1. Implement missing-cell/raw-preserving transfer and receipt-safe
   post-assembly stages.
2. Add the canonical pool orchestrator and sha-pinned CLI with deterministic
   manifest/diagnostic paths and no tolerance knobs.
3. Convert the legacy CLI to a deprecated shim while preserving the helper
   imports its known consumer needs.
4. Add synthetic full-path tests, changelog fragment, focused verification, and
   the external review worklog.
