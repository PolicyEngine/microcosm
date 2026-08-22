# Progress: replacement scorecard

## State

The `replacement-scorecard` lane is active on top of `de5ca6aa` (post-#741
`origin/main` plus two lane journal commits). The task: one head-to-head
scoring path for the live US incumbent and the not-yet-built 25% bundle-mode
candidate, score the incumbent now, and leave the owner the exact candidate
command. Three earlier attempts died (network loss / codex quota); their
salvage refs all reduce to one 1306-line scorer draft, which this session
verified mechanism-by-mechanism and is rewriting where its premises were
stale.

No pool build, push, gate change, threshold change, or band change has
occurred. `ps ax | grep build_us_multispine_pool` showed no live build before
any scoring step.

## Done

- `uv sync --all-packages --extra us` completed normally in this session
  (earlier sandbox venv-cloning notes are historical).
- Verified every symbol the salvaged draft imports and every cited mechanism
  against the code this session: compile surface (five inputs, no membership
  flags), dropped/skipped detection, sqrt–concept-budget–50/50 loss weights,
  capped weighted-MAPE aggregate, relative-error rule, attribution row keys,
  battery registries (131 + 1 joint) and receipt keys, pool-manifest
  authentication chain, `terminal_gates` manifest shape.
- **Corrected the incumbent identity** (see `_LANE-NOTES.md` "incumbent
  identity corrected"): live policyengine.py is 5.0.3 (2026-08-21), whose
  bundled manifest resolves the US default dataset to
  `policyengine/populace-us` revision
  `populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z`,
  filename `populace_us_2024.h5`, SHA-256 `48b9d479...` — microcosm's own
  buildp sparse artifact, not enhanced_cps_2024 (that was 4.15.0-era). Local
  cached bytes hash-verified.
- Re-classified the terminal battery for the real incumbent: provenance
  columns present, but zero `acs`-channel rows (channels: asec 22,200 @ clone
  0, puf_tax_detail 35,040 @ clone 1), so ASEC-vs-ACS comparisons are
  inapplicable on observed evidence.
- Established that the incumbent H5 predates CD vintage provenance attrs, so
  the scorer needs a recorded legacy waiver path for entity H5s missing the
  attrs (strict when present).

## Next

- Rewrite `tools/score_us_release_head_to_head.py` accordingly; add the
  contract/determinism/fixture tests; ruff + affected suites green; commit.
- Re-check the build queue, run the incumbent scoring (RSS < 20 GiB), commit
  `experiments/replacement_scorecard/incumbent_48b9d479.{json,md}`.
- Record the owner's candidate command and the comparison doctrine in
  `_LANE-NOTES.md`; run the full workspace suite; update `FINAL_REPORT.md`.
