# PR #747 defensive audit progress

## State

- In progress on local audit branch `review/pr-747-audit`; the PR branch remains unmodified.
- Reviewing cached PR head `86f55741081fa3fb5e3c55234e3c8dc7ff77c777` against merge base `7b90bb1882b0248d751a64bf817ec127e5c42a47`.
- GitHub and the default uv cache are unavailable in the sandbox; the requested sync also failed from a writable cache because PyPI DNS is blocked. No repository build will be run.
- Exact diff inventory: 27 commits, 51 files, 9,965 insertions and 3,922 deletions (the largest churn is generated `source_stages.json`).

## Done

- Read `CLAUDE.md` and the GitNexus PR-review workflow.
- Attempted the requested PR fetch and `gh pr view`; both failed because the sandbox cannot resolve GitHub.
- Identified the cached matching remote-tracking head `origin/uk-spine-assembly-686`, checked it out as local `pr-747`, then branched before adding audit artifacts.
- Retried `uv sync --all-packages --extra us` with `/tmp/uv-cache-review747`; resolution reached the locked `joblib==1.5.3` download and stopped on blocked DNS.
- Located a complete sibling-worktree environment whose `uv.lock`, workspace metadata, and package metadata are byte-identical to the PR; it will run only targeted tests with this worktree's sources first on `PYTHONPATH`.
- Inventoried the entire PR file/commit surface and assigned independent receipt, doctrine, and code/test audits.

## Next

- Read the full diff, every added receipt/register/fixture, and relevant doctrine.
- Verify suspicious claims with targeted tests and independent receipt checks.
- Write `REVIEW-747.md` with a merge verdict and code-cited evidence.

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
