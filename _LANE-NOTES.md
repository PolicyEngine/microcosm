# US reviewed-exclusion machinery lane notes

## Scope and invariants

- Branch/worktree: `battery-excl-machinery` in the assigned worktree only.
- Port reviewed-exclusion machinery, not an exclusion decision.
- The committed US register must remain empty; only the owner may approve an
  actual exclusion.
- Frozen bands, thresholds, seeds, folds, and comparators are immutable.
- Builds use at most the 1% sample with seed 578, remain off-chain, and never
  touch `logbook-pending-chain.txt`.
- Every mechanism statement added below will carry a source file and line
  citation.

## Evidence read

- `experiments/battery_burndown/ADJUDICATION.md` from the read-only
  `microcosm-arm-split` worktree.
- Its `adjudication.json`, filtered to the two
  `REVIEWED-EXCLUSION-CANDIDATE` rows and the reviewed-exclusion support
  record.

## Mechanism record

- The production US battery accepts only the frame and optional tail manifest,
  then invokes the comparison evaluator with no exclusion policy
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11366-11378`).
- A boolean comparison with zero weighted incidence on both origins appends a
  dead-comparison failure and records `status=dead_comparison`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11819-11849`).
- The QBI reconciliation deliberately sets both clone-0/native SSTB flags
  false on the ASEC-compatible support role
  (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1246-1264`),
  while clone provenance defines index 0 as the ASEC role and donor-detail
  clones as the PUF role
  (`packages/microcosm-build/src/microcosm/build/us_runtime/support_provenance.py:370-473`).
- The UK schema-2 template requires exactly five receipt fields, canonical
  dates, and an expiry strictly after approval
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/weighted_integrity.py:264-349`),
  and its strict loader rejects duplicate/non-finite JSON and any schema drift
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/weighted_integrity.py:545-624`).
- The UK degenerate gate applies only in-force entries, fails stale/expired/
  premature entries, and reports dormant ones
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/terminal_gates.py:294-433`).
- US stacked gate details are snapshotted and evaluator-sealed before
  `GateReport` copies them into a manifest
  (`packages/microcosm-build/src/microcosm/build/gates.py:142-177,204-309`).
- The pool writes that same gate payload into the immutable terminal receipt
  and final manifest aliases
  (`tools/build_us_multispine_pool.py:3383-3384,3421-3484,4057-4087`).

## 1% before receipt

- Source: `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/smoke-f001-r10/logbook-receipts/populace-us-2024-stacked-f001-s578-asec1688-acs15316-20260813T041921Z-23bea977/terminal-gates.json`
- SHA-256: `96f68ddac5c833cb3c6a3eb6ea3ceb0cf04a5dcbbf92ef565814ed368a6ce5`.
- `person/puf_tax_itemization/sstb_self_employment_income_would_be_qualified[clone_0]#dead`:
  failed; ASEC/ACS incidence 0/0; nonzero rows 0/0.
- `person/puf_tax_itemization/business_is_sstb[clone_0]#dead`: failed;
  ASEC/ACS incidence 0/0; nonzero rows 0/0.
- An empty register cannot honestly change either measured value or waive
  either failure. The measurable after result must be authenticated policy and
  exact-criterion reporting while both unapproved checks remain red.

## Work log

- 2026-08-19: initialized the lane journals before implementation.
- 2026-08-19: the required uv sync is currently environment-blocked: the
  managed sandbox denies writes to the default cache, and the writable-cache
  retry cannot resolve PyPI DNS. No dependency or lockfile change was made.
- 2026-08-19: completed the code-cited design audit and recorded the immutable
  1% baseline receipt.
