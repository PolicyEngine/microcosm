# Progress

## State

Populace #550 focused rebuild is complete locally on
`ssi-507-takeup-stabilizer`, based on merged #549 at `f964356`. The work
follows the integration verdict's KEEP/DROP/fix-in-A lists and preserves
#548's collect, merge, write, then single-batched-raise terminal flow.

## Done

- Confirmed the clean `f964356` baseline, read `CLAUDE.md` and the complete
  integration adjudication, and committed this journal before implementation.
- Attempted the GitNexus exploration workflow; its index/tools were not
  available in this session, so the audit used current source/test call sites
  and `stabilizer-v1-archive` directly.
- Advanced SSI take-up diagnostics to schema 4 with a version-history note
  explaining schema 3's ambiguous floor-blind/floor-aware prior semantics.
- Restricted schemas 2 and 3 to legacy target/capacity/floor seeds, allowed an
  absent schema-3 phase marker, rejected explicit non-final legacy phases, and
  retained release-final plus full-gate requirements for schema 4.
- Added the frozen-support identities digest beside the SSI assignment digest
  in the target-frame checkpoint identity and #217 reform-vector cache
  context, wired directly from the selection source.
- Added failing-first regressions for the new schema and selection behavior,
  the real Build O attempts 2/3 floor-aware arithmetic receipt, and an isolated
  final-integrity Bernoulli-law failure that reaches the written calibration
  receipt before the single terminal batch raises.
- Added the requested towncrier fragment. `_band_prior`, per-target knobs,
  reconcile loops, dedicated post-write raises, and AST ordering pins remain
  untouched.
- Applied Ruff formatting to the four touched Python files.
- Passed the exact requested pytest command over
  `test_us_ssi_take_up.py`, `test_us_fiscal_refresh_builder.py`, and
  `test_us_plan.py`; `uv run ruff check .` and `git diff --check` are clean.

## Next

- Review and open the focused #550 PR when authorized. Nothing was pushed
  during this work.
- Treat green PR tests as code-contract evidence only; restricted-data release
  certification remains a separate human-run workflow.
