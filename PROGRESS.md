# PR #747 defensive audit progress

## State

- Round 2 began on 2026-08-25 under the user's explicit stale-head warning. No
  round-1 conclusion is being carried forward without re-verification.
- Required live PR head:
  `76e39f9b65e498a4361ce708c786c27954cfb93d`. It is not yet present in the
  shared object database or refs.
- Three exact fetch attempts have failed before transfer because the shell
  environment has no DNS configuration and cannot resolve `github.com`.
  Therefore no round-2 finding has been evaluated yet and the required
  `uv sync --all-packages --extra us` has not started.
- Work remains isolated on `review/pr-747-audit`; no PR branch was modified,
  committed to, pushed, built, published, or promoted.

## Done

- Read `CLAUDE.md` and the `gitnexus-pr-review` instructions again for round 2.
- Verified the worktree and existing round-1 journal were clean before the
  fetch attempts.
- Verified both the full live SHA and local ref `pr-747-r2` are absent, rather
  than silently falling back to cached head `86f55741`.
- Confirmed `gitnexus status` reports this repository is not indexed; no
  repository analysis/indexing was started while the exact live source is
  unavailable.

## Next

- Acquire and check out live head `76e39f9b`, verify it byte-exactly, then run
  the required US-extra sync before evaluating any finding.
- Rebase the audit artifacts onto a separate round-2 audit branch rooted at
  that exact live head, preserving this journal and the round-1 report.
- Re-verify every round-1 finding as `SURVIVES` or `RESOLVED`, inspect all new
  commits/receipts/fixtures, run only scoped tests and probes, and update
  `REVIEW-747.md` with the fresh verdict.

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
