# ACS housing-universe + reported-income composition (root integration candidate)

Lane journal for the local, reversible integration candidate combining the two
frozen author lanes on top of the root integration head.

## State

Branch `us-acs-income-composition-20260905`, worktree
`_worktrees/microcosm-us-acs-income-composition-20260905`, created from the
exact root integration head **a9aa9356d0ab3d6ac1b9131680eab8d4b0194f1e** ("Add
authenticated SIPP December source evidence"), which was clean when the branch
was cut. `origin/main` is **c1b8324183819cf91511c002bf724a2676524e87**; the base
is 113 ahead and 0 behind it.

Inputs, both frozen and left untouched:

- ACS: `_worktrees/microcosm-us-acs-hu-fable-followup-20260905` at
  **91b3ee51e75141c2bb821a30d873cdf3e770cccf**, declared base
  **a276a3c93aed561dd08f7f209b3303fbae2e2c44**, which is an ancestor of the
  integration head.
- Income: `_worktrees/microcosm-us-income-parser-review-fixes-20260905` at
  **0b507da14bebfaab7253f327a2e8acbd98d52d16**, base
  **a9aa9356** — the integration head itself.

## Done

- Read AGENTS.md/CLAUDE.md, fetched `origin/main`, compared both lanes to it.
- Established that the integration head's own 20 commits since `a276a3c9`
  touch none of the three shared files, so the ACS delta is portable forward.
- Took 22 disjoint files by exact blob and hand-composed the three shared ones
  (`graph_sources.py`, `graph_implementation.py`, the inventory), keeping both
  stage registrations and both stage-specific CSV identity profiles.
- Recomputed the `graph_sources.py` dependency contract against the composed
  bytes with the repository's own `_dependency_contract`: unchanged.
- 384 pure cases green, zero failures/errors/skips; 274 loaded modules match
  the head's git blobs; 18 sealed income files byte-identical;
  `ci_test_groups.py --verify` ok; ruff check/format and `git diff --check`
  clean. Composed at **6b26cbbc**, tree clean.

## Next

- Root: run the installed-engine `test_us_graph_asec_prepared.py` under a
  separate assignment; it was deliberately not run here.
- Root: decide whether the successor review's six residual non-blocking items
  are addressed before or after this candidate moves.

## Scope fences

No push, no merge, no publication, no genuine source run, no model run, no
network. Root PROGRESS/journal files from both lanes are excluded. The 18 sealed
income legacy files and the integration head's SIPP/PUF/exporter code are
preserved. This is a reversible local integration candidate, not source-fitness
or release approval.
