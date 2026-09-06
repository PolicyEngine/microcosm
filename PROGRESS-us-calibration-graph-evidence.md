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
