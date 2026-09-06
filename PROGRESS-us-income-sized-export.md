# Sample-size-aware income-v3 export successor — progress

Child worktree of the accepted income-v3 exporter head. Scope: add a separately
named, versioned **successor** export profile whose semantics are closed and
stable but whose cohort size is measured from verified outputs rather than
hard-coded, so the genuine 505-household run can be represented — while the
original strict 15-household profile keeps working byte-for-byte.

Local code and invented fixtures only. This lane exported nothing, installed
nothing, ran no model and wrote no approval.

## State

- Worktree: `_worktrees/microcosm-us-income-sized-export-20260906`
- Branch: `us-income-sized-export-20260906`
- Base (accepted parent): `5c276925c723e17af175709b5602ee7ace32d530`
- Parent worktree `microcosm-us-income-v3-export-review-fixes-20260906`
  preserved, clean, untouched at that commit
- Started and completed: 2026-09-06

## Done

1. Created the child worktree from the accepted parent head; copied the parent's
   `.venv` and repointed every editable `.pth` at this worktree, so tests here
   exercise this worktree's code and not the parent's. Baseline: 226 passed.
2. Read both genuine run receipt sets and computed the reconciliation set
   directly: every equality the sized profile enforces was verified to hold in
   the 15-household run and the 505-household run before it was written.
3. `graph_evidence_income_sized.py` — profile
   `us_asec_income_v3_five_node_sized_v1`, binding schema
   `microcosm.income_v3_sized_root_export_binding.v1`. Imports the reviewed
   profile and edits nothing in it; inherits every semantic table by reference.
4. Explicit named producer adapters (`income_v3_first_genuine`,
   `income_v3_measured_505hh`) covering the exact field and output differences
   between the two producers. The root binding names the adapter; the run's own
   recorded schema must be that adapter's.
5. Measured scope with a `Measurement` ledger: every count read from recorded
   outputs, required to agree across each independent recording, reconciled
   against the arithmetic, and required to equal root's declared scope.
6. 96 sized tests at two invented shapes (9 households equal cohorts on the
   first producer; 41 households with unequal cohorts 14/14/13 on the second),
   including a byte-identity assertion on the reviewed profile and the base
   exporter.
7. Doc `docs/us-income-sized-graph-evidence-export.md`, cross-linked from the
   reviewed profile's doc, plus a changelog fragment.
8. 322 tests green (226 reviewed unchanged + 96 new); ruff clean;
   `tools/ci_test_groups.py --verify` ok with the new suite registered in
   `shared-spec` rather than defaulted.
9. Verified against the genuine 505 run without exporting: the full validation
   chain passes and stops at the candidate-writer boundary.

## Defect found and fixed

Requiring two independent recordings of every required entity count would have
refused a correct run — the first producer records the tax-unit count exactly
once. The rule now applies to households and persons, the two counts everything
else is sized by; the rest reconcile against root's declared scope.

## Next

- Root reviews this head, then the binding proposal in
  `R/us-income-sized-export-505hh-root-binding-PROPOSAL.json`, then exports and
  serves the genuine 505 run. Command in
  `R/us-income-sized-export-INTEGRATION.md`.
- Nothing here approves anything.
