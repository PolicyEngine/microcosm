# Progress

## State

Populace #462 critical-row loss alignment is in progress on
`loss-contract-alignment`, starting from clean `origin/main` at `c3e378a`.
The builder and publisher consume one shared critical-target register, and the
solver now applies a configurable critical-row loss boost after all existing
normalization. The default and CLI override are recorded in staging telemetry
and failure-safe calibration diagnostics. No target values have changed.

## Done

- Confirmed the worktree is clean on `loss-contract-alignment`.
- Confirmed `HEAD` exactly matches `origin/main` at `c3e378a`.
- Replaced the prior merged-task progress record with this task's baseline.
- Read the populace #462 adjudication thread and the local Build N attempt 5/6
  diagnostics; confirmed builder gates passed while publication rejected the
  national medical-deduction row at +21.2% / +20.78% against 15%.
- Extracted the US critical fit requirements into a dependency-light shared
  `populace.data` module used by both builder and publisher.
- Added the missing itemized, SALT, and medical classes to the builder,
  including the publisher's family/role alias matching and no-incumbent-escape
  semantics. The existing Table 1.4 blanket is derived from the same source.
- Threaded the target registry into production gate evaluation so live Table
  1.2 and Table 4.3 role aliases are covered, not only the five finite names.
- Added an anti-drift superset/tolerance test plus focused alias and medical
  gate regressions.
- Verified the populace-data contract suite (65 passed) and focused builder
  critical/release-gate tests (18 passed).
- Added `US_CRITICAL_TARGET_LOSS_MULTIPLIER = 5.0` and applied it once to every
  shared-contract match after concept-budget, amount/count, and optional family
  normalization, with no post-overlay renormalization.
- Matched loss rows by full compiled identity (`name@period`), shared semantic
  selectors, and the Table 1.4 blanket while excluding congressional-district
  layouts; overlapping selectors cannot multiply a row twice.
- Added `--critical-target-loss-multiplier` with positive-finite validation for
  adjudication runs, and recorded the effective value in calibration-stage
  telemetry and `calibration_diagnostics.json`'s build record.
- Added unit coverage proving the critical row is exactly 5x its normalized
  base weight, non-critical/CD rows are unchanged, and grouped concept budgets
  were normalized before the overlay. Added CLI, diagnostics-record, and main
  orchestration assertions.
- Verified the focused loss/CLI/diagnostics/orchestration group (15 passed).
- Ran the complete populace-data package: 132 passed, with one pre-existing
  optional-engine skip.
- The first complete builder run exposed four manifest tests whose registry
  doubles omitted the real registry's `.specs` surface. Updated those test
  doubles for the new registry-aware gate signature and reran the file: 131
  passed.

## Next

- Run the required final Ruff fix/format pass and diff audit.
- Write `FINAL_REPORT.md` and perform the final clean-worktree audit.
