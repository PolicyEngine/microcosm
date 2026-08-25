# Progress: defensive post-merge audit of PRs #747 and #743

## State

Audit in progress on local branch `postmerge`, checked out from merge commit
`2807e99cc11d1a8c69a0886992ae21a533719958`. The GitHub REST interface
independently confirmed that this is current `main`, that PR #747 merged it at
2026-08-25 11:45:37 UTC, and that its second parent is PR head `76e39f9b`.
All requested #747 round-one surfaces survive on that tree. No build,
publication, promotion, push, or review-branch mutation has occurred.

The required `uv sync --all-packages --extra us` was attempted. The default uv
cache is outside the writable sandbox, and the writable-cache retry resolved
the 123-package lock but could not download `joblib==1.5.3` because external DNS
is unavailable. Focused probes will use an already compatible environment only
where its exact limitations can be stated.

## Done

- Read `CLAUDE.md` and the GitHub/GitNexus PR-review instructions.
- Preserved the pre-existing untracked `PR747-R2-COMMIT.txt` and
  `PR747-R2-LIVE.diff` evidence files unchanged.
- Attempted `git fetch origin`; recorded the DNS failure rather than treating
  remote-tracking refs as fresh.
- Created `postmerge` at the available `origin/main` merge commit.
- Attempted the mandated all-package US-extra sync with both default and
  task-local writable uv caches.
- Confirmed with `git range-diff` that the 27 audited #747 patches are
  patch-equivalent after rebase and that the only added PR patch is
  `76e39f9b`, which repairs entity-table scoping only.
- Ran the six focused #747 age/NHS/parity/register/payload files successfully
  on the sibling environment whose `uv.lock` SHA-256 exactly matches this
  checkout, with this checkout first on `PYTHONPATH`.
- Refreshed the production E6 probe on merge tree `2807e99c`: 189 of 400
  top-coded people moved to age 85+, 67 to age 90+, permutation identity stayed
  true, and all six stored NHS columns disagreed with recomputation.
- Refreshed adversarial parity probes: missing and value-1 person/benunit counts
  and an opposite-direction water delta were still signed; strict band `0.9`
  accepted a `-0.776937` share delta; a missing weighted-total key plus an
  anonymous/missing identity still returned strict parity.
- Verified `76e39f9b` does reject a wrong entity scope while leaving the
  quantitative register failures intact.
- Reconfirmed the committed receipt says 24 stages while production runs 25
  ending in `age_tail`, with no tracked acceptance receipt, and reconfirmed the
  stale 0.6761 incumbent figure against the pinned 0.674658 reference.

## Next

1. Rerun every finding-specific probe on the current #743 head and map Vahid's
   four second-pass findings for overlap/supersession.
2. Append cited round-2 verdicts to both review reports, commit and update only
   `review/pr-747-audit` and `review/pr-743-audit`, then write the output report.
