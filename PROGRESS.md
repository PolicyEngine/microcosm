# US SIPP December housing source projection — lane journal

Branch `us-sipp-housing-projection-20260906`. Base `origin/main`
`c1b8324183819cf91511c002bf724a2676524e87`, fast-forwarded to the frozen parent
`ab10f77854b7b9e8057a07dbec40d15c10b96297` (`origin/main` is an ancestor of the
frozen head; the frozen head is 124 commits ahead, so the fast-forward is exact).

Scope: housing metadata, a pure housing measurement kernel, and an executable,
versioned source projection **recipe**. No genuine source extraction, no fit, no
transfer, no publication. Root runs the extraction after reviewing this lane.

## State

In progress.

## Done

- Confirmed the base/frozen-head relation and created this worktree.
- Located and hash-verified the cached official 2023 SIPP data dictionary.

## Next

- Transcribe the seven required fields into an explicit recipe.
- Add the housing measurement kernel, tests and documentation.
- Write the bounded first-32-SSUID extraction proposal (not executed).
