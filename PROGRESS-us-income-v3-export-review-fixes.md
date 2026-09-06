# income-v3 graph export — Fable 5.1 review fixes

Child worktree of the root exporter head. Scope: close the actionable items in
`us-income-v3-graph-export-fable51-review.md` (REQUEST_CHANGES) narrowly, without
exporting or installing anything.

## State

- Worktree: `_worktrees/microcosm-us-income-v3-export-review-fixes-20260906`
- Branch: `us-income-v3-export-review-fixes-20260906`
- Base (root exporter head): `1056bd56ac4c483d7afb03d9c1e255a244eb1f95`
- Upstream base: `origin/main` = `c1b8324183819cf91511c002bf724a2676524e87`
- Started: 2026-09-06

## Done

1. Read the Fable 5.1 review, the exporter, the base exporter, both test modules,
   the export doc, the two source contracts, the accounting and engine kernels,
   the root pins document, the accepted root output review and the dashboard
   reader's attachment checks. Recorded the pre-change hashes.
2. Finding 1 — publish `census_ccpiu_annual_2024`, the price basis the kernel's
   own contract names, with the contract digest beside it, and require the
   accounting node to record that contract. CPI-U-RS is gone from the bytes.
3. Finding 2 — `_node_receipt_bindings` binds the engine node's
   `evaluation_sha256` and the accounting node's `accounting_sha256` when the
   run records them. `accounting_sha256` is the payload digest, traced to
   `graph_asec_income.py`.
4. Finding 3 — each graph phase's `reported_income` summary is validated against
   the summary contract and required to equal the direct block field for field.
5. Finding 4 — `_resealed_manifest_mutation` restores the content body, the
   content key and each phase receipt's recorded key, so node cases reach
   `_nodes`. Every case asserts its own refusal text. `INPUT_PRODUCERS` makes
   "accounting never consumes the engine" an enforced rule.
6. Finding 5 — focused negatives for the accounting layout, the carried housing
   cells, the lineage roster and the graph summary.
7. Doc updated, including the reader boundary and the closed identifier roster.
8. 167 income-v3 + 59 four-node + 74 writer tests green; ruff clean repo-wide;
   `tools/ci_test_groups.py --verify` ok.
9. Verified against the accepted genuine run, without exporting: every changed
   validator accepts it.

## Next

- Root re-reviews this code, then re-exports to a new namespace.
- Reader extension is separate work and is documented, not attempted.
