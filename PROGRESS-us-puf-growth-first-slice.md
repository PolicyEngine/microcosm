# PROGRESS — US raw-PUF growth transform, first slice

Slice-scoped journal. The shared root `PROGRESS.md` is untouched by this branch.
Per repo CLAUDE.md, root journals are history, not state: read git/GitHub for truth.

Branch `us-puf-growth-first-slice-20260905`, worktree
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-puf-growth-first-slice-20260905`,
forked from root integration `9ff943e6896ee6c487c853e4da96ccdfd7d96ed5` (unchanged by this branch).

## State (2026-09-05)

Implemented, adversarially reviewed, all findings answered, tested green,
documented, committed. Not pushed, no PR, nothing wired into a build, a source
pin, or a release. Root inspects independently.

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

- Added `packages/microcosm-build/src/microcosm/build/us_runtime/puf_growth.py`:
  the pure contract (roles, growth kinds, sign branches, variable-row/year-column
  factor levels, authority) and the transform.
- Added `packages/microcosm-build/src/microcosm/build/us_runtime/puf_growth_graph.py`:
  boundary / money / design-weight nodes and kernels.
- Added `packages/microcosm-build/tests/test_us_puf_growth.py` (29 tests) and
  `packages/microcosm-build/tests/test_us_puf_growth_graph.py` (6 tests). All 35 pass.
- Added `docs/us-puf-growth-first-slice.md` and
  `changelog.d/20260905-us-puf-growth-first-slice.added.md`.
- `ruff check .` clean repo-wide; `tools/ci_test_groups.py --verify` reports
  `verification=ok`; both new files land in `us-p` and `rest`, never `[defaulted]`.

- Ran a five-dimension adversarial review (transform arithmetic, contract
  loopholes, graph semantics, honesty, coverage-by-mutation). It raised 40
  findings; all 40 were answered — fixed, or restated where the defect was an
  overclaim in prose. Two would have shown up in CI or hidden a regression: a
  `sys.modules` country-engine guard that fails in the engine lane, and a
  fresh-store parity assertion that compared node-key-derived ids rather than
  values.
- Final: 67 tests (56 contract, 11 graph), `ruff check .` clean repo-wide,
  `tools/ci_test_groups.py --verify` = `verification=ok`.

## Next

- Hand to root for independent inspection. Root composes; this branch does not
  wire the stage into any build, pin, or release path.
- The obligations that remain before any of this touches real data are in
  `docs/us-puf-growth-first-slice.md`, "Remaining obligations".

## Scope fences (from the authorizing brief)

No genuine PUF reads, no model or resource generation, no fit, no network asset download,
no release/candidate mutation, no push, no PR, no messages to other agents. Invented factors are
non-authority fixtures. Not in this slice: QBI/199A, Forbes-tail, disclosure disaggregation,
raw-artifact authentication, genuine factor generation.
