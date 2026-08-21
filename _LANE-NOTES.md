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

## 2026-08-21 — runtime surface unified

- The public compiler has no CD membership option and routes IRS SOI, ACS-CD,
  and PEP-CD facts unconditionally
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-1010,2299-2390,2588-2625`).
- The row-level doctrine is unchanged: CD rows compile into the registry, but
  `_soi_taxable_interest_control_key_from_fact` rejects CD record sets as
  national controls
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:1478-1510,1726-1744`).
- Builder, fiscal scorer, state scorer, and ACS-local compilation all load the
  packaged source-to-current CD crosswalk by default; the production source
  aliases are active unconditionally
  (`tools/build_us_fiscal_refresh_release.py:1486-1497,8319-8350,11226-11230`;
  `tools/score_us_fiscal_targets.py:383-426`;
  `tools/score_us_state_files.py:313-345`;
  `tools/build_us_acs_local_release.py:141-170`).
- The diagnostic JCT deletion switch and its target-profile-gate bypass were
  deleted from all release/scorer entrypoints. The active registry is now the
  compiled registry in the builder and both scorers
  (`tools/build_us_fiscal_refresh_release.py:8435-8464`;
  `tools/score_us_fiscal_targets.py:415-432`;
  `tools/score_us_state_files.py:335-352`).
- The generated calibration contract declares `national`, `state`, and
  `congressional_district` in one `geography_layers` list and requires CD
  inclusion; there is no default-layer fork
  (`tools/us_bundle_generation/contracts.py:1322-1340`;
  `packages/microcosm-build/src/microcosm/build/spec_engine/schema/calibration.schema.json:35-68`).
- The parity generator compiles with the canonical crosswalk unconditionally,
  and the regenerated manifest records 32 compiled / 52 reviewed families
  (`tools/build_us_target_parity_manifest.py:617-655,689-729`;
  `packages/microcosm-build/src/microcosm/build/us/target_parity_manifest.json:3-12,524-527`).
- This deletes the dead scorer opt-in described by
  [microcosm#569](https://github.com/PolicyEngine/microcosm/issues/569) and the
  regime knob superseded by the one-surface decision in
  [microcosm#449](https://github.com/PolicyEngine/microcosm/issues/449#issuecomment-5002607353).
- Validation: the affected 10-file suite completed to 100% with exit 0 after
  the change; Ruff, Python byte compilation, and `git diff --check` pass. No
  build, push, chain operation, or pending-chain edit occurred.

## 2026-08-21 — parity doctrine and row-level invariant

- `irs_soi.congressional_district_2022` is a red-line compiled family, so the
  anti-rot validator refuses any future downgrade to a reviewed exclusion
  (`packages/microcosm-build/src/microcosm/build/us_runtime/release_target_parity.py:88-123,579-586`).
- The shipped manifest entry is `compiled`, has no exclusion classification,
  reason, evidence, or fence, and states that there is no local-versus-national
  surface. Its header counts are pinned to the parsed 32 compiled and 52
  reviewed families
  (`packages/microcosm-build/src/microcosm/build/us/target_parity_manifest.json:3-12,524-527`;
  `packages/microcosm-build/tests/test_release_target_parity.py:290-317`).
- The always-compiled SOI test exercises CD, state, and national rows
  (`packages/microcosm-build/tests/test_us_fiscal_targets.py:177-356`). The
  taxable-interest doctrine test additionally asserts that the CD aggregate is
  present in the registry, then proves it never supplies the rebase control
  metadata when a true Pub 1304 national control exists
  (`packages/microcosm-build/tests/test_us_fiscal_targets.py:2539-2625`).
- Validation: the standard 10-file affected suite completed to 100% with exit
  0; Ruff and `git diff --check` pass.

## 2026-08-21 — artifact-scale leak removed

- The hidden split was the compiler's per-run support-exclusion mapping: a
  caller could delete otherwise compiled source rows for one artifact. That
  parameter and its dynamic-dispatch branch are gone; the compiler now has
  exactly five inputs, none related to artifact size, sparsity, record count,
  support, inclusion, or diagnostic target deletion
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-1005,2069-2090,2287-2300`).
- The release tool no longer parses or loads an artifact-specific exclusion
  file, passes no membership override into compilation, and reports only the
  standing surface-wide source exclusions
  (`tools/build_us_fiscal_refresh_release.py:897-923,7770-7815,8363-8376,11192-11198`).
  The obsolete
  `experiments/build_j_recert/sparse_zero_support_exclusions_buildj.json` was
  deleted and its shell/caller plumbing removed.
- The shared `US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` registry remains: it is a
  single source-row doctrine applied identically to all artifacts, not a scale
  input (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:554-660,2287-2295`).
- The identity regression compiles one CD-bearing fact set twice under nominal
  57,240-record sparse and 337,704-record dense labels; both the full `specs`
  tuple and content-addressed registry `version` must match, and the exact
  compiler signature is pinned
  (`packages/microcosm-build/tests/test_us_fiscal_targets.py:644-696`).
- Release/fiscal-scorer/state-scorer signature sets are pinned, and the release
  parser rejects all three deleted membership options
  (`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:52-97`;
  `packages/microcosm-build/tests/test_us_state_files_scorer.py:26-40`).
- Validation: six new identity/signature/parser tests pass; the standard
  10-file affected suite completed to 100% with exit 0; Ruff and
  `git diff --check` pass. No build ran.
