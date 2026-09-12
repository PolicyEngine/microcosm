# PR triage lane journal (2026-09-12)

Read-only triage of every open PR in PolicyEngine/microcosm. GitHub is read
only; no tracked file is edited; commits land on the local ref
`triage/journal-20260912` via plumbing so the shared worktree HEAD never moves.

## State

- Output: `~/PolicyEngine/_recovered/scratch-backup/893/triage/out-pr-triage.md`
  (the `-o` path, confirmed from the subfleet run meta) and
  `~/PolicyEngine/_recovered/scratch-backup/893/triage/pr-dispositions.json`.
- Scratch: session scratchpad holds `open_prs.json`, `merged_prs.json`,
  `closed_unmerged_prs.json`, `pr/<n>.json`, `checks/<n>.txt`.

## Done

- Listed 43 open PRs, 444 merged PRs (305 since 2026-07-01), 16 closed-unmerged.
- Pulled full `gh pr view --json` for all 43; ran `gh pr checks` for all 43.
- Fetched all 43 heads to `refs/heads/triage/<n>`; SHAs verified against GitHub.

## Next

- Compute merge-bases, diffs vs main, #893 overlap, patch-id supersession.
- Fan out per-PR evidence + disposition; adversarially verify every close.
- Write both output files.

## 2026-09-12 ~08:00 ET — mechanical evidence complete

- Per PR: merge-base vs its base branch, ahead/behind main, changed-file set,
  overlap with #893's 404-file set, `git cherry` patch-equivalence (0 of 43 PRs
  have any patch-equivalent commit on main), added-file presence on main
  (raw or via the populace->microcosm rename), review threads (0 for all 43),
  linked-issue states (99 numbers resolved).
- `packages/populace-build/` is absent on main; PRs #153–#629 (17 PRs) still
  target it.
- #893 split into 96 per-commit file lists (`Stage integration layer 01..11`).
- Base stacks: #339 on `medicaid-chip-take-up` (its PR #334 merged 7/09);
  #532<-#534<-#535<-{#536,#537}; #536<-#538; #893<-#901; #904<-#906.
- Next: per-PR judgment fan-out (workflow), adversarial verification of every
  close-class disposition, then the report.

## 2026-09-12 ~08:25 ET — judgment fan-out running; owner spot checks

- Workflow `wf_5af23347-412`: 43 evidence agents → lens-specific refuters
  (close-class PRs get unique-work, superseder and dependents refuters;
  keep-class get a supersession hunter; MERGE-READY gets a readiness check)
  → one coherence critic.
- Owner spot checks so far: #752 is superseded (main merged #756
  "Bump policyengine-us to 1.819.0", commit 05d254aa2, 2026-08-24; uv.lock and
  the pin test on main both read 1.819.0). #817's `requires_us` markers are
  absent on main. #642 (SLD), #718 (campaign naming), #888 (explorer.py),
  #629 (`reform_dict`) symbols are absent on main. #339's commit is not on
  main although its base branch merged as #334. Current aggregate `ci-ok`
  job dates from 404e45777 (2026-08-25); PRs showing only
  `test (3.13/3.14)` + `wheels (3.13/3.14)` ran under the old workflow.
- Note: the sibling issue-triage lane checked out `issue-triage-20260912` in
  this shared worktree (HEAD moved 0c3f4f651 → 3e56e503b); this lane's
  commits stay on `triage/journal-20260912` via plumbing and are unaffected.
