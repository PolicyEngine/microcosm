# Progress: defensive post-merge audit of PRs #747 and #743

## State

Audit in progress on local branch `postmerge`, checked out from merge commit
`2807e99cc11d1a8c69a0886992ae21a533719958`. The GitHub REST interface
independently confirmed that this is current `main`, that PR #747 merged it at
2026-08-25 11:45:37 UTC, and that its second parent is PR head `76e39f9b`.
All requested #747 round-one surfaces survive on that tree. No build,
publication, promotion, push, or review-branch mutation has occurred.

The current PR #743 head is exactly the round-one audited object
`74b4d768f7f7c83eb0593464ceb0e2a7c81ec154`; all five P0/headline findings
survive. The prompt's named seam commits `96e73047`, `971a9699`, and
`e54e546b` are ancestors of that object, not later repairs. This was checked
against the refreshed Git `FETCH_HEAD`, the remote-tracking ref, and GitHub's
REST branch response.

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
- Checked out the exact current #743 object detached and ran the five-file
  seam/scorer/materializer slice plus the exact expected-abort national-stage
  regression successfully. An initial invocation named the materializer test
  in the wrong shard and exited before collection; the corrected path passed.
- Refreshed all five #743 P0 probes: the runbook path still has no resolver;
  all 397 production measures remain slash-named and the scorer raises with
  none persisted; arbitrary H5 bytes retain trusted default labels without
  hashes; three children still reduce to one across eight bindings; the
  `uk-national-calibration` pipeline still derives undeclared scope
  `uk/national` and resolves its predecessor after artifact writes; and
  release posture still accepts a custom five-entry, ownerless exclusion
  surface while excluding `uk_target_surface` from its battery.
- Read Vahid's four second-pass findings from PR comment 5408202783. None
  supersedes a round-one finding: clone-weight collapse is separate; band-edge
  selection is adjacent to but distinct from boolean-count reduction; the
  unreachable `knows()` fence is adjacent to but distinct from the runbook's
  missing resolver; and the no-mutation concern is resolved as a non-finding
  by `UKFrameTargetAdapter` copying every source table at construction.

## Next

1. Append cited round-2 verdicts to both review reports, commit and update only
   `review/pr-747-audit` and `review/pr-743-audit`, then write the output report.
