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
