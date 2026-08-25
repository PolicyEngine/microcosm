# residual 1% verify build: failure-line diff vs the pkg3 base

Cold 1% build (sample seed 578) on this branch at the serial host queue,
2026-08-25. Physical (mirror-deduplicated) battery failure lines from
pool.gates.json:

- pkg3 base: **114**
- this branch: **113**
- greened: **1** — `person/source_operator_weeks_unemployed/weeks_unemployed[clone_0]/positive` incidence
  (was ratio 0.0314, out of [0.8, 1.25]; the SHA-pinned no-build replay
  predicted 1.00067 and the build confirms the leg is green)
- introduced: **0**
