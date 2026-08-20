# F1 continuation r4 progress

## State

- Work is local on `spec-engine-f1`; nothing has been pushed and no sample
  build has run in this continuation.
- Deliverable 7 is complete. Commit `ef036572` retargeted the active lineage
  consumers to compiler outputs, `823a0100` reopened a whole-column closure
  defect, and the exact-cell correction is present at HEAD `5875be22`.
- The correction preserves authority independently in each compiler predicate
  space. A typed column may therefore have graph, early-family, and take-up
  segments when their exact scoped cells do not collide; the dashboard does
  not invent cross-space supersession.
- The source correction was swept into concurrent main-lane commit `5875be22`
  while both lanes shared the index. Its D7 files and evidence are being
  closed out separately in a prefixed D7 handoff commit; the shared commit is
  not rewritten.
- Deliverables 4/5/6/8 and certification remain owned by the main F1 lane.
  The host ceiling is 20 GiB RSS, and this split-out remains capped at 15 GiB.

## Done

- Read the charter, owner ruling, lane journal, RFC evidence, compiler seams,
  and held #697 closure artifacts before changing consumers.
- Audited tests and tools for held `us_imputation_lineage.yaml`, the stale
  392-column f025 inventory, authored lineage classes, closure helpers,
  segment surfaces, and the dashboard. No active held filename, loader, or
  class consumer remains.
- Retargeted the dashboard and all four production-shaped test consumers to
  `compile_spec(load_bundle(...))`, `CompiledSpecIR.resource()`, typed
  inventory, producer graph, and compiler predicate registries. Explicit
  generation-0 parity oracles and synthetic low-level tests remain authored.
- Corrected the derived closure to retain all 20 graph/family shared columns
  because their exact atoms are disjoint. Added fail-closed checks for unknown
  contracts, empty or duplicate atoms, peer-surface cell collisions, and
  missing typed-column closure.
- Independently established the current projection: 173 typed contracts, 241
  raw graph write segments, 170 final graph-owner segments/763 cells, 152
  typed graph segments/134 columns/735 cells, 48 early-family segments/cells,
  14 take-up leaves/13 columns/26 cells, and 214 combined segments/809 unique
  scoped cells.
- Ran all four D7 modules after the correction: 70 tests passed in 407.29
  seconds. The exact projection regression also passed alone in 85.94 seconds;
  the emitter completed in 43.10 seconds; Ruff, formatting, bytecode, and
  whitespace checks passed.
- Obtained two read-only independent approvals. One reconstructed the exact
  closure directly from compiler IR without importing dashboard helpers; the
  other emitted byte-identical JSON under two `PYTHONHASHSEED` values.
- Kept the historical root journal and unrelated main-lane files intact.

## Next

1. No further deliverable-7 implementation or verification is pending.
2. The main F1 lane may consume the append-only D7 handoff while continuing
   its deliverables 4/5/6/8 work and owner-run certification gates.
