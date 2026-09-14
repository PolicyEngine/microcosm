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

## Next (run 3)

- Refresh `gh pr view`/`gh pr checks` for all 50; refetch all 50 heads.
- Recompute mechanical evidence against the new `origin/main` and new #893 set.
- Rebuild packets for all 50; fan out evidence + adversarial verification.
- Write `out-pr-triage.md` and `pr-dispositions.json`.
