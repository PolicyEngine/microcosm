# US runtime module classification completion

Scope: `packages/microcosm-build/tests/test_us_spine_blindness.py` plus one
changelog entry. Child of exact `9ece4cad`. Root worktree
`microcosm-us-launch-current-main-20260905` was NOT edited (verified: clean,
still at `9ece4cad`, identical `us_runtime` tree digest
`02b95376…`).

## State
COMPLETE for the assigned check. One pre-existing failure reported, not fixed.

## Done
- Reproduced the reported failure in this child: `test_us_spine_blindness.py:3313`
  `::test_us_runtime_module_classification_is_complete`, same 21 names as
  `R/us-9ece-spine-blindness.log`.
- Read all 21 modules and traced production reachability. None is imported by
  `multispine_pool.py`, `late_producer_dag.py`, `us_late_producer_registry.py`
  or `graph.py`; all run pre-assembly or outside the pool pipeline.
- Established the repo's operator boundary as **post-assembly population
  treatment** — not "has `register_us_*_kernels`" and not "declares `Owned(...)`",
  since `graph_cps_carried`, `graph_prior_year`, `graph_person_preparation`,
  `graph_early_preparation` and `graph_geography` have both and are already
  classified non-registry.
- Classified all 21 in `_OTHER_US_RUNTIME_MODULES` in six commented groups.
  That list grants no source-spine column access and no scanning exemption.
- Full-file regression, engine-free and network-blocked: **498 tests, 1 failed,
  0 errors, 0 skipped**. The reported test now passes; the fail-closed
  `test_runtime_classification_rejects_a_new_unreviewed_module` still passes.
- `tools/ci_test_groups.py --verify` → `verification=ok`; registration unchanged.

## Reported, not fixed (out of scope)
`test_runtime_population_operators_are_source_spine_blind`
(`test_us_spine_blindness.py:3416`) fails on 7 of these modules with fail-closed
**opacity** findings — numpy boolean-mask subscripts, dict lookups keyed by a
loop variable, byte slices. No source-spine column is named. Confirmed
**pre-existing**: it fails identically against the unmodified `9ece` test file;
root's `-x` run stopped before reaching it. Fixing it needs a scanner predicate
change, a source-owner exemption, or a production edit — all out of scope, so it
was reported with path/line/evidence instead.

## Next (root's call)
- Independently inspect the classifications and the diff before applying to I.
- Re-adjudicate `puf_growth_graph.py` if the intended boundary is "owns population
  cells" rather than "post-assembly treatment" — it is the one close call.
- Decide the route for the pre-existing opacity failure.

Durable report: `R/us-runtime-classification-completion-REPORT.md` / `.json`
(`R` = `~/architecture-reviews/microcosm-20260904`).
