# Combined clone review corrections — 2026-09-06 historical handoff

Continues accepted development clone `ae26cb0cf` in a fresh worktree created
from fetched `origin/main` and fast-forwarded to that source. The original
reports and journals are preserved.

The invented semantic comparator now excludes raw entity IDs and replaces
person memberships with each referenced group's declared source/channel/role
key. Tests separately establish that renumbering every ID preserves semantic
comparison while raw-ID comparison detects every change, and that changing a
membership to a different source group still fails semantic comparison.

Ten cases reproduce the independently checked explicit-weight refusals: four
group entities with and without equal explicit person weights, equal person
weights alone, and the forbidden all-zero vector. Another test demonstrates
refusal when halving the smallest subnormal loses mass. The direct operator,
executor and weight core are unchanged.

Comments and documentation now scope floating receipt totals outside the
BITWISE output promise, describe the actual operator role-count divisor, and
state the missing assembly metadata integration boundary. CLAUDE.md points
descriptively to downstream declared nodes rather than forbidding continuation.

The source-proof plugin, reproducible runner, logs, JUnit and final report are
archived at `/Users/maxghenis/architecture-reviews/microcosm-20260904/` under
`clone_review_source_proof.py` and `us-combined-clone-review-polish-*`.
Those artifacts report timing separately from JUnit, console and runner sources,
and record the final clean Git identity. No genuine population, cache,
checkpoint or microdata was read; no operator/core behavior or runtime package
was changed, and nothing was pushed or published.
