# Progress: candidate-chain smoke CD-vintage provenance defect

Date: 2026-08-24

Branch: `candidate-25pct-runbook`

## State

STOP decision reached: current main has a stacked-pool/release producer-consumer
contract gap, not an omitted stage-1 or stage-2a runbook argument. The release
always selects a CD-vintage crosswalk and preflights the input H5, while the
production stacked-pool surface cannot assign household districts, stamp the
required provenance, or publish the table-format node the guard probes. The
detailed code-cited finding is committed in
`experiments/candidate_25pct/cd_vintage_provenance_defect.md`.

Per the requested boundary, `run-candidate.sh` and `--dry-run` remain unchanged;
no guard or vintage code was weakened; no H5 was altered; no pool/release build
or smoke rerun was launched; and `smoke_r2.md` was not created.

## Done

- Read `CLAUDE.md`, including the PR-CI/certification boundary and root-journal
  conventions.
- Read the GitNexus debugging and exploration skill instructions selected for
  this code-path investigation.
- Confirmed the worktree is clean on `candidate-25pct-runbook` at
  `7c14a3ba1186dc2a2ba6125ee0d3000aaf140345`.
- Identified `FINAL_REPORT.md` as the existing repository output report for
  this branch; it will be replaced only after the investigation is resolved.
- Traced the release CD path end to end: unconditional packaged crosswalk,
  strict preflight before Ledger compilation, later target-fact translation,
  and diagnostic-only CD gate handling.
- Traced the production stacked-pool parser, source boundary, operator graph,
  publication manifest, nullable writer, and the separate PUF-support producer.
- Verified the stacked writer deliberately emits fixed-format entity groups
  while the release guard recognizes only a compound `household/table` node.
- Read-only inspected the existing 1% H5 and release log: both required root
  attrs, the semantic household CD column, and the physical table node are
  absent; the packaged crosswalk digest itself is correct.
- Compared the relevant runtime files with `origin/main` at `7b90bb18`; they are
  byte-identical.
- Ran five focused parser/default/guard/writer tests; all passed.
- Documented the exact missing current-main producer, provenance, H5-reader,
  and integration-test wiring in the defect report.

## Next

1. Commit this defect finding and progress state as a coherent investigation
   step.
2. Replace `FINAL_REPORT.md` with the concise stop report, run repository
   hygiene checks, and commit the finalized journal/report.
