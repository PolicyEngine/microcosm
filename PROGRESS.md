# Progress: defensive post-merge audit of PRs #747 and #743

## State

Audit and output are complete on local branch `postmerge`, which began at PR
#747 merge commit `2807e99cc11d1a8c69a0886992ae21a533719958`.
Final live `main` is `f9d3b4838c137e113e87452ccdeaa734b11733d2`;
GitHub compare proves its sole later commit changes one word in `DESIGN.md`, so
every #747 finding surface and probe input remains byte-identical. All requested
#747 round-one surfaces survive. No build, publication, promotion, or
production-pointer mutation occurred.

The current PR #743 head is exactly the round-one audited object
`74b4d768f7f7c83eb0593464ceb0e2a7c81ec154`; all five P0/headline findings
survive. The prompt's named seam commits `96e73047`, `971a9699`, and
`e54e546b` are ancestors of that object, not later repairs. This was checked
against the refreshed Git `FETCH_HEAD`, the remote-tracking ref, and GitHub's
REST branch response.

The required `uv sync --all-packages --extra us` was attempted. The default uv
cache is outside the writable sandbox, and the writable-cache retry resolved
the 123-package lock but could not download `joblib==1.5.3` because external DNS
is unavailable. Focused probes used an already compatible exact-lock
environment with its limitations recorded.

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
- Updated `REVIEW-747.md` on local `review/pr-747-audit` through commit
  `6b01dda54389621efe7fd8c90e281d7945223e7c`, including six owner-ready issue
  drafts and the final-live-main check.
- Updated `REVIEW-743.md` on local `review/pr-743-audit` through commit
  `6743d119d54f65cec3160bf7247d77a0d7a5f09b`, including the five-verdict table
  and Vahid overlap analysis.
- Re-read live `main` and PR #743 immediately before finalization. Main had
  advanced only by the unrelated one-word `DESIGN.md` commit; #743 remained at
  `74b4d768`.
- Attempted three authorized Git pushes limited to the two review branches;
  all failed because the shell could not resolve `github.com`. The connector
  publication gate was not approved, so remote refs remain unchanged and no
  unauthorized branch was pushed.
- Replaced `FINAL_REPORT.md` with the complete survives/resolved tables,
  second-pass mapping, issue drafts, and verification/delivery record.

## Next

1. The owner should publish local commits `6b01dda5` and `6743d119` to their
   matching review branches once GitHub DNS is available; no force push is
   required against the refs observed during this audit.
2. File the priority E6/NHS issue first and do not rely on or promote the
   merged UK spine until its final-age identity contract is repaired and the
   exact repaired 25-stage candidate is receipted.
