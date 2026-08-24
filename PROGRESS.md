# Progress: candidate-chain smoke CD-vintage provenance defect

Date: 2026-08-24

Branch: `candidate-25pct-runbook`

## State

Investigation started from the clean requested branch head `7c14a3ba` with the
PolicyEngine-US 1.819.0 lock. The failed 1% smoke pool and release state are
external evidence only until the congressional-district vintage provenance
path has been traced end to end. No runbook change or smoke rerun has occurred.

The authorized decision boundary is strict: patch only supported stage-1 or
stage-2a runbook wiring. If current main provides no supported way to satisfy
the release guard, stop and report the missing main-branch wiring instead of
weakening the guard or changing the vintage module.

## Done

- Read `CLAUDE.md`, including the PR-CI/certification boundary and root-journal
  conventions.
- Read the GitNexus debugging and exploration skill instructions selected for
  this code-path investigation.
- Confirmed the worktree is clean on `candidate-25pct-runbook` at
  `7c14a3ba1186dc2a2ba6125ee0d3000aaf140345`.
- Identified `FINAL_REPORT.md` as the existing repository output report for
  this branch; it will be replaced only after the investigation is resolved.

## Next

1. Discover or attempt the GitNexus graph workflow, then trace the release CD
   path, pool writers, current tests, and #733/#735/#741-era history in source.
2. Decide whether supported runbook wiring exists; either implement and test it
   or stop with an exact code-cited main-branch defect report.
3. If supported, rerun only the authorized 1% smoke stages, commit
   `experiments/candidate_25pct/smoke_r2.md`, and finalize this journal and
   `FINAL_REPORT.md`.
