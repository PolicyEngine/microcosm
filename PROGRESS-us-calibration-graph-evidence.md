# Demographic calibration graph development

Started from frozen integration ab10f77854b7b9e8057a07dbec40d15c10b96297 after
fetching origin/main c1b8324183819cf91511c002bf724a2676524e87 (124 ahead, zero
behind). The parent was clean and remains unchanged. This branch owns only the
calibration seam, tests, documentation and changelog.

The first eight direct/contract tests pass. A subsequent full graph test exposed
that the executor requires its original-design cap declaration; the new kernel
now separates the solver's incoming-importance cap from that executor cap. The
three-node fixture exercises unequal anchors, real calibration, full diagnostic
parity and a fresh-process warm replay. Nine tests passed at this checkpoint.
No genuine source or reference was read for these tests and no national registry
was activated. The final focused group passed 84 tests, including 16 new cases, with zero failures, errors or skips. Ruff checks and the tracked CI grouping verification pass. The test process recorded 249 loaded Microcosm modules. Independent Fable review remains pending.

## Fable correction pass

Fable 5.1 reviewed committed 497ff338 independently. Fixed normative-node comparison so descriptive fields work, derived artifact schema from the payload constant, included both cap anchors in the diagnostic artifact, refused non-positive incoming weights and skipped targets, and added monetary-binding source bytes to implementation identity. Meaningful new tests cover descriptions, zero-weight refusal, an original-design cap that rejects a solver-accepted result, and missing original design anchors.

Root executed 88 focused tests (20 new-module cases plus 68 existing calibration cases), zero failures/errors/skips. After test-only formatting, all 20 new-module cases were rerun on final formatted bytes; cold/direct parity and a separate fresh-process warm run with forbidden kernels passed. The two earlier correction-fixture failures (wrong ArtifactType test attribute and an invalid same-kind test setup) are preserved in external logs; neither was a production failure. Ruff check/format pass. No genuine inputs or activated targets.

Canonical initial Fable full+summary remain under /tmp/policyengine-command-runs/f514308cd017/demographic-calibration-497ff338-review-*. Incremental independent review of corrected commit follows.
