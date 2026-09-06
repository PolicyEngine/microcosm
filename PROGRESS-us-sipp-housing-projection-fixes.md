# US SIPP December housing projection — review-fix lane journal

Branch `microcosm-us-sipp-housing-projection-fixes-20260906`. Worktree created
from fetched `origin/main` `c1b8324183819cf91511c002bf724a2676524e87` and
fast-forwarded to the reviewed head
`725d9c54d5436f64ec2c49e9ccab496c1bfede1b` (`origin/main` is an ancestor of that
head, so the fast-forward is exact). The author worktree
`_worktrees/microcosm-us-sipp-housing-projection-20260906` is untouched.

Scope: close S1–S6 from the independent Fable 5.1 review of the December SIPP
housing projection, with consequential tests and accurate docs. No genuine
source extraction, no fit, no transfer, no publication, no push.

## State

Starting. Review context, capture and root reproductions read; code read.

## Done

- Read the review context, the Fable 5.1 capture and the root r2 reproductions.
- Created this worktree, confirmed a clean head at the reviewed commit, and
  confirmed hooks are disabled (`core.hooksPath=/dev/null`).
- Read `sipp_housing_measurements.py`, `sipp_housing_projection.py`, the frozen
  `sipp_measurements.py` / `sipp_december_source.py` internals they reuse, the
  test file, the doc, and the v1 extraction proposal and self-check.

## Next

- S1 strict assistance classification, S2 v2 proposal, S3 receipt selection
  evidence, S4 implementation binding, S5 topcode-marker diagnostics,
  S6 the enumerated missing tests.
