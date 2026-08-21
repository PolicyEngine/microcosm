# One-target-surface lane notes

## 2026-08-21 — baseline and doctrine

- Branch: `one-target-surface`, starting at `2c7a7218` (`origin/main`).
- User doctrine: all US calibrated artifacts compile one target surface;
  geography is a constraint dimension, while artifact scale changes only L0 /
  record count.
- The current split is explicit in
  `packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py`:
  `compile_us_fiscal_target_registry` accepts
  `include_congressional_district_targets`, forwards it through dynamic target
  dispatch, and uses it to omit SOI and ACS congressional-district facts.
- The target profile independently gates CD-to-state hierarchy reconciliation
  on the same flag in
  `packages/microcosm-build/src/microcosm/build/us/fiscal_target_references.json`.
- The SOI taxable-interest rebase doctrine remains distinct: CD aggregate rows
  are processing-window subsets and never act as national controls, as recorded
  in `_rebase_stale_soi_taxable_interest_distributions` and pinned by
  `test_stale_soi_taxable_interest_never_uses_congressional_district_controls`.
- [microcosm#449](https://github.com/PolicyEngine/microcosm/issues/449#issuecomment-5002607353)
  explicitly names deletion of the flag as the one-surface outcome;
  [microcosm#569](https://github.com/PolicyEngine/microcosm/issues/569)
  records that the scorer's opt-in path is dead.
- Environment: default-cache sync failed because the sandbox cannot write
  `~/.cache/uv`; clean-cache sync then failed because network/DNS is disabled.
  The sibling `microcosm-spec-engine` checkout has the identical `uv.lock`
  (`895535...`) and a complete Python 3.14 environment, so its `.venv` was
  cloned copy-on-write. An offline editable reinstall still requires missing
  build-isolation metadata; test commands therefore use `UV_NO_SYNC=1` and put
  every current-worktree package `src` directory first on `PYTHONPATH`.
- GitNexus: repository analysis produced `.gitnexus/lbug`, then registration
  failed on the sandboxed `~/.gitnexus/registry.json`; dependency coverage is
  being checked with repository-wide `rg` plus focused tests.
- Build discipline: no calibration build has run; the off-chain / <=1% rule is
  intact and `logbook-pending-chain.txt` has not been touched.
- Validation baseline: `uv run pytest -q` advanced without a failure for
  1,136.97 seconds, then was interrupted while an unrelated PUF-QRF test was
  waiting on a subprocess
  (`test_puf_qrf_chain.py::test_primary_qrf_rejects_every_stale_schema_version`).
  Per-commit checks use the affected US target/compiler tests; the complete
  workspace suite will run against the final tree. `uv run ruff check .`
  passed.
- Affected-suite baseline: the target/compiler/parity/builder/scorer/spec set
  reached 100% with no failure in 349.59 seconds. `/usr/bin/time -l` itself
  exits 1 in this sandbox because `sysctl kern.clockrate` is forbidden (the
  same wrapper also exits 1 around `true`); later memory receipts use Python's
  child-resource accounting instead.
