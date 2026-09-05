# PROGRESS — US raw-PUF growth transform, first slice

Slice-scoped journal. The shared root `PROGRESS.md` is untouched by this branch.
Per repo CLAUDE.md, root journals are history, not state: read git/GitHub for truth.

Branch `us-puf-growth-first-slice-20260905`, worktree
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-puf-growth-first-slice-20260905`,
forked from root integration `9ff943e6896ee6c487c853e4da96ccdfd7d96ed5` (unchanged by this branch).

## State (2026-09-05)

Setup and reconnaissance complete; implementation not started.

## Done

- Confirmed root worktree HEAD `9ff943e6896ee6c487c853e4da96ccdfd7d96ed5`, clean, ahead 71 of
  `origin/main` (`c1b8324183819cf91511c002bf724a2676524e87` after `git fetch origin --prune`).
- Created this dependent worktree/branch at that exact commit.
- Read in full: `R/us-puf-producer-provenance.md`, `R/us-other-donor-price-basis-audit.md`
  (R = `/Users/maxghenis/architecture-reviews/microcosm-20260904`), and Microcosm issue #530.
- `uv sync --all-packages --locked --offline` (no `us`/`uk` extras): `policyengine_us`,
  `policyengine_uk`, `policyengine_core` are all absent from this environment by design.
- Probed `import microcosm.build.us_runtime`: succeeds in 37.75 s and loads **no** `policyengine*`
  module, so its `__init__` does not trigger installed-PolicyEngine ABI initialization here.

## Next

- Design the per-field growth contract against the ASEC current-money precedent.
- Implement the registry + pure transform + tiny Frame/graph demonstration.
- Write `docs/us-puf-growth-first-slice.md` with the explicit remaining obligations.

## Scope fences (from the authorizing brief)

No genuine PUF reads, no model or resource generation, no fit, no network asset download,
no release/candidate mutation, no push, no PR, no messages to other agents. Invented factors are
non-authority fixtures. Not in this slice: QBI/199A, Forbes-tail, disclosure disaggregation,
raw-artifact authentication, genuine factor generation.
