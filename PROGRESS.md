# PR #747 defensive audit progress

## State

Round-2 post-merge verification is complete against final live `main`
`f9d3b4838c137e113e87452ccdeaa734b11733d2`. PR #747 merged as
`2807e99cc11d1a8c69a0886992ae21a533719958` with PR head `76e39f9b`; the sole
later main commit changes one word in `DESIGN.md`, so every probed executable,
resource, receipt, and register byte is identical at the final tip. All five
numbered round-one findings survive; the entity-table subpart of finding 2
alone was resolved by `76e39f9b`. The priority production defect is a fresh E6
reproduction in which final age-tail redistribution makes all six stored NHS
columns disagree with their final-age recomputation.

`REVIEW-747.md` now carries the full post-merge verdict table, refreshed probes,
resolving-commit analysis, and six owner-ready issue drafts. No build,
publication, promotion, production-pointer change, or commit to the merged PR
branch occurred.

## Done

- Read `CLAUDE.md` and the GitHub/GitNexus review instructions.
- Attempted the required fresh shell fetch and US-extra sync; recorded the
  sandbox DNS/cache failures. Used GitHub REST for live refs and a sibling
  exact-lock environment for focused tests only.
- Proved with `git range-diff` that all 27 audited patches are patch-equivalent
  after rebase and the sole added PR patch is entity-scope repair `76e39f9b`.
- Re-ran the production NHS allocator → persisted outputs → age-tail → E6 path:
  189/400 moved to age 85+, 67 to 90+, and all six NHS stored columns failed
  while permutation identity remained true.
- Re-ran arbitrary/missing count, opposite-direction water, widened strict
  band, and incomplete/unbound weighted-total probes; every fail-open survived.
- Rechecked 24-vs-25 stage evidence, absence of the named tracked receipt, and
  stale 0.6761-vs-0.674658 incumbent evidence.
- Ran the focused parity/register/payload/age/NHS test slice with exit 0 and
  validated the updated report with code-line citations.
- Re-read live `main` immediately before finalization. It had advanced by one
  commit to `f9d3b483`; GitHub compare and commit evidence show exactly a
  one-word `DESIGN.md` edit and no finding-surface change.

## Next

1. The owner should file the priority E6/NHS issue first, then the five proof
   completeness/enforcement issues drafted in `REVIEW-747.md`.
2. Do not rely on or promote the merged spine until the E6 defect is fixed and
   the exact repaired 25-stage candidate has identity-bound acceptance.

---

## Historical round-1 audit (stale head `86f55741`)

### State

- Audit and report completed on local audit branch `review/pr-747-audit`; the PR branch remained unmodified.
- Reviewed cached PR head `86f55741081fa3fb5e3c55234e3c8dc7ff77c777` against merge base `7b90bb1882b0248d751a64bf817ec127e5c42a47`.
- GitHub and the default uv cache were unavailable in the sandbox; the requested sync also failed from a writable cache because PyPI DNS was blocked. No candidate/repository build was run.
- Exact diff inventory: 27 commits, 51 files, 9,965 insertions and 3,922 deletions (the largest churn is generated `source_stages.json`).
- Historical verdict was HOLD: the committed acceptance evidence predated the value-changing 25th stage and SPI persisted-value changes, the final stage order reproduced an E6 NHS identity failure, and the strict parity instrument had independently reproduced fail-open paths.

### Done

- Read `CLAUDE.md` and the GitNexus PR-review workflow.
- Attempted the requested PR fetch and `gh pr view`; both failed because the sandbox cannot resolve GitHub.
- Identified the cached matching remote-tracking head `origin/uk-spine-assembly-686`, checked it out as local `pr-747`, then branched before adding audit artifacts.
- Retried `uv sync --all-packages --extra us` with `/tmp/uv-cache-review747`; resolution reached the locked `joblib==1.5.3` download and stopped on blocked DNS.
- Located a complete sibling-worktree environment whose `uv.lock`, workspace metadata, and package metadata are byte-identical to the PR; it ran only targeted tests with this worktree's sources first on `PYTHONPATH`.
- Inventoried the entire PR file/commit surface and assigned independent receipt, doctrine, and code/test audits.
- Read the full diff and both experiment records, and checked every changed committed register, resource, pin, generated manifest, and fixture relevant to the swap claim.
- Established commit ordering: the experiment records end at `a7336d31`; candidate-changing `19799f37` subsequently adds `age_tail` and SPI persisted-value changes; `86f55741` only re-pins the roster-derived release manifest and test.
- Ran 250 focused tests against this worktree's sources: 249 passed and 1 skipped across parity, signed-register, age-tail, identity, reference, terminal-gate, weighted-integrity, source-stage, and country-spec coverage.
- Reproduced the final-order E6 failure with production functions and the committed age-band resource: 188/400 synthetic top-coded persons moved to 85+, and all six stored NHS columns then disagreed with E6's recomputation from final ages.
- Reproduced four parity fail-opens: arbitrary or missing signed entity counts remain unsigned-clean; an opposite-direction water regression is signed; a 0.70 unsigned share delta disappears at `--share-band 0.9`; and an omitted weighted-total key is reported but not failed.
- Confirmed no terminal threshold, criticality, reviewed-exclusion register, target membership, Logbook rule, publication path, or promotion side effect changed.
- Ran `git diff --check` over the exact PR range successfully.
- Wrote the ranked, code-cited verdict and minimum closure conditions to `REVIEW-747.md`.

### Next at round-1 close

- María should hold the merge until the final 25-stage candidate is rebuilt and receipted and the strict comparator/register fail-opens are closed; then rerun this audit against the new head.

---

## Historical PR-head journal (preserved verbatim)

# Progress: PolicyEngine-US 1.819.0 lock bump

## State

Complete and ready for the owner to open the PR. The lock resolves
`policyengine-us==1.819.0` and `policyengine-core==3.31.0`; all required
compatibility repairs, generated contracts, attested identities, package test
shards, generator checks, and Ruff are green. The code-cited mechanism audit,
identity inventory, compatibility note, and validation receipts are in
`_LANE-NOTES.md`.

No pool-consumed variable was removed without a verified successor, so no
owner question is pending. No pool/release build, push, gate change, threshold
change, tolerance change, or band change occurred.

## Done

- Started and committed this standing progress log at `514964d4` after
  rebasing the lane start onto current `origin/main`.
- Ran the ordered initial sync attempt, documented the sandbox cache/network
  limits, and completed the unchanged-lock and upgraded-lock all-package US
  syncs from exact official artifacts in a task-local cache.
- Ran `uv lock --upgrade-package policyengine-us`. The complete version
  movement is exactly `policyengine-core 3.26.11 -> 3.31.0` and
  `policyengine-us 1.764.6 -> 1.819.0`; NumPy 2.4.6 and Torch 2.12.0 did not
  move.
- Verified upstream variable reality in the installed PE-US 1.819.0 package,
  adapted Microcosm's input ownership and consumer guards, and retained
  fail-closed certified-dataset version checks. Each mechanism is cited in
  `_LANE-NOTES.md` under “verified upstream compatibility repairs.”
- Regenerated the release-input, parity, source/take-up, engine-ABI, spec,
  seed, coverage, and golden identities with repository tools. The final
  46-value inventory is in `_LANE-NOTES.md`; the final commit body carries the
  exhaustive old-to-new mapping.
- Added the requested owner-facing PE-US 1.764.6-to-1.819.0 compatibility note
  from the installed release changelog, including major SNAP, receipt,
  OBBBA-follow-through, cash/health/housing, tax, and new-program changes.
- Passed every package test: 7,213 passed / 77 skipped / 0 failed. Calibrate,
  data, fit, and frame ran as individual package shards. The build inventory
  was proven as a complete disjoint 4,161-item + 2,180-item partition after
  the otherwise-green single process retained more than the binding memory
  ceiling; accepted peaks were 12,596,384 and 13,363,984 KiB, both below 15
  GiB. This uses the repository's fresh-process shard rationale
  (`.github/workflows/test.yml:24-34`) without changing any assertion or model
  behavior.
- Reproduced 163 required release inputs / 7 reviewed exclusions / 41 reform
  probes; 32 compiled parity targets / 52 reviewed exclusions; US spec SHA-256
  `3189d90dec95c8ea7090e41b5283fa52b1e6855bed4a776dfa02820f2bd11c62`;
  and 42,096/42,096 configuration fields plus 40/40 inventory checks.
- Passed repository-wide Ruff and `git diff --check`; wrote the final handoff
  to `FINAL_REPORT.md`.

## Next

The owner opens the PR. No push or additional build is required from this
lane.
