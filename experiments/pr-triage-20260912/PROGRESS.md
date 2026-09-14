# PR triage lane journal (2026-09-12 → 2026-09-14)

Read-only triage of every open PR in PolicyEngine/microcosm. GitHub is read
only; no tracked file is edited; commits land on the local ref
`triage/journal-20260912` via plumbing so the shared worktree HEAD never moves.

## State

- Output: `~/PolicyEngine/_recovered/scratch-backup/893/triage/out-pr-triage.md`
  (the `-o` path, confirmed from `brief-pr-triage.md`) and
  `~/PolicyEngine/_recovered/scratch-backup/893/triage/pr-dispositions.json`.
- Scratch: session scratchpad holds `open_prs.json`, `merged_prs.json`,
  `closed_unmerged_prs.json`, `pr/<n>.json`, `checks/<n>.txt`, `packets/<n>.md`.
- **Run 3 (2026-09-14) is the live attempt.** Runs 1–2 (2026-09-12) died on
  model limits; their evidence is salvaged under `scratchpad/prior/` and the 16
  completed evidence-agent verdicts under `scratchpad/prior_results/`.

## Done (runs 1–2, 2026-09-12 — historical, re-verified in run 3)

- Listed 43 open PRs, 444 merged PRs (305 since 2026-07-01), 16 closed-unmerged.
- Pulled full `gh pr view --json` for all 43; ran `gh pr checks` for all 43.
- Fetched all 43 heads to `refs/heads/triage/<n>`; SHAs verified against GitHub.
- Per PR: merge-base vs its base branch, ahead/behind main, changed-file set,
  overlap with #893's then-404-file set, `git cherry` patch-equivalence (0 of 43
  PRs had any patch-equivalent commit on main), added-file presence on main
  (raw or via the populace->microcosm rename), review threads (0 for all 43),
  linked-issue states (99 numbers resolved).
- `packages/populace-build/` is absent on main; PRs #153-#629 (17 PRs) still
  target it.
- #893 split into 96 per-commit file lists.
- 16 of 43 evidence agents returned before the limit hit.

## Done (run 3, 2026-09-14)

- Salvaged runs 1-2 scratch + the 16 completed verdicts.
- `git fetch origin`: main is now `18271b28d` (merge of #904). #912 (Amendment
  20) and #904 both merged since run 1, so **every** run-1 overlap/merge-base
  number is stale and is being recomputed.
- Open-PR set is now **50, not 43**: new since 9/12 are #914 #916 #918 #921
  #923 #924 #926 #927 #928; #904 and #912 left the set by merging.
- #893 itself moved: 404 changed files on 9/11 -> 691 on 9/14, so its overlap
  set is recomputed from the new head.

### Mechanical evidence complete for all 50 (2026-09-14 ~18:50 ET)

- `gh pr view --json` refreshed for all 50; `mergeable`/`mergeStateStatus` now
  computed for every one (run 1 left 35 at UNKNOWN). Main moving turned many
  MERGEABLE rows CONFLICTING: 22 of 50 are now CONFLICTING/DIRTY.
- All 50 heads refetched to `refs/heads/triage/<n>`; every local SHA verified
  equal to `headRefOid`.
- `gh pr checks` + head-attached `statusCheckRollup` for all 50. CI eras:
  10 LEGACY (only the retired `test/wheels` quartet - stale green),
  11 NONE (no checks on the head at all), 29 CURRENT.
- Recomputed per PR against `origin/main=18271b28d`: merge-base, ahead/behind,
  net changed files, overlap with #893's **new 641-file** set, `git cherry`
  patch-equivalence (again **0 of 50** have a patch-equivalent commit on main),
  added-file presence on main raw/under the populace->microcosm rename, and the
  count of paths still under the pre-rename `packages/populace-build/` tree.
- #893 is now **rebased onto current main** (merge-base == `18271b28d`,
  0 behind, 281 commits ahead, 641 files in the net diff; GitHub reports 691
  changedFiles - the gap is GitHub's per-commit union vs the net diff).
  Its 11 named `Stage integration layer NN` commits are indexed, plus 270 later
  commits; 877 distinct files mapped to the commit that touches them.
- #906 was **retargeted to main** after #904 merged and is now MERGEABLE/CLEAN
  with `ci-ok` SUCCESS - the brief's "#906 is based on #904's branch" is stale.
- 90 of the 95 issue numbers referenced by PR bodies resolved; 5 (#1791 #9306
  #9311 #9346 #9405) are not issues in this repo (they are policyengine-us /
  policyengine-uk numbers).
- Packets rebuilt for all 50 at `scratchpad/packets/<n>.md`.

### Judgment fan-out + independent owner spot checks (2026-09-14 ~19:10 ET)

- Workflow `wf_6f60b8fc-b4d`: 50 evidence agents -> lens-specific refuters
  (close-class PRs get unique-work / superseder / dependents; MERGE-READY gets a
  readiness audit + supersession hunt; KEEP-BLOCKED-ON gets a blocker-reality
  check + supersession hunt) -> a per-PR adjudicator -> two coherence critics.
- Independent owner spot checks recorded in `scratchpad/owner_spotchecks.md`
  (verified, not delegated): #752 superseded by 05d254aa2/#756; #817, #749,
  #642, #888, #167, #413 unique on main; #153 superseded by #904's
  `dft_bus_value_anchors.json` + #730; #181 by #714's `cash_isa`; the QBI stack
  splits (#532's v1 port duplicated by #893's `puf_qbi_model.py`, but #534-#538's
  v2/v3 SSTB-crosswalk / SCF / SOI content exists nowhere); #918 adds amendments
  **25 and 26** (main is already at 24, not 20) and its `weight_update.py` is
  byte-identical to #901's while #893 lacks the file entirely.
- **#914 is the one real fold**: 12 of its 20 files are byte-identical to #893,
  including `calibrate/target_snapshots.py` and its test, and #893 commit 136 is
  literally "Merge the completed #908 snapshot slice". Its red CI is verified as
  stale country-bundle spec digests
  (`test_spec_engine_country_bundles.py:86` spec_sha256 mismatch x3).
- Byte-compared every 893-overlapping file of all 29 overlapping PRs: only #914
  has substantive identity. Filename overlap alone folded nothing.

## Next (run 3)

- Finish the fan-out, reconcile against the owner spot checks.
- Write `out-pr-triage.md` and `pr-dispositions.json`.
