# Progress: microcosm #722

## State

The adjudicated B1 and B3 re-verdict findings are closed locally on
`passive-pass-through-722` as of 2026-08-22. B1 closes in `bd537c42`; B3 closes
in `92dd3568`. The earlier merge and remediation history below remains the
branch record. Work remains offline: no fetch, push, PR mutation, publication,
or promotion is in scope, and clean-runner CI remains the suite arbiter.

## Done

- Read the binding adversarial verdict before inspecting or changing the
  branch.
- Re-read `CLAUDE.md` and confirmed the PR-CI/certification and root-journal
  contracts for this remediation pass.
- Attempted the GitNexus debugging workflow; no GitNexus query/context tools
  are configured in this session, so direct source and call-path analysis is
  the documented fallback.
- Confirmed the branch, detached pre-approved merge worktree, merge base, and
  current `origin/main` identities without network access.
- Resolved the only three merge conflicts with the approved typed country
  resource rows, both independent multispine imports, and one measured runtime
  graph pin. The three resolved file blobs match the pre-approved detached
  index exactly.
- Ran the post-merge source-blindness graph test itself with repository
  conftests disabled because the pre-sync environment does not yet contain
  main's new `jsonschema` dependency. The single registered tool reaches the
  pinned 67-module graph (base 64, branch +2, main +1).
- Traced the country-package closure failure to
  `tools/generate_us_bundle_from_constants.py::LEGACY_RESOURCE_PATHS`; the
  loader's `_assert_file_closure` compares that emitted manifest with package
  files.
- Committed the real merge of current `origin/main` as `d9f19c66`; its second
  parent is `2c7a7218` and it is not a rebase or squash.
- Declared the three passive/repeal JSON resources on
  `LEGACY_RESOURCE_PATHS`, completed the full generator run, and ran the exact
  offline `uv sync --all-packages --all-extras` equivalent against the locked
  environment.
- Traced the remaining constants-adapter failure to the passive contract being
  present only as generated pool code. Added it to the typed pipeline schema,
  compiler identity contract, and stacked-authority projection, then regenerated
  the bundle at spec SHA-256
  `11b5f830141d007b53131f6c863d15a6be10787fc6d88aeb147317d2c1f593e0`.
- Independently reviewed the changed field-usage claims: 62 new pointers total
  (29 authored, 33 resolved), all accounted for by the three typed manifest
  rows and the typed passive pipeline/pool-code contract.
- Regenerated the F0 evidence report with exact 41,442/41,442 field coverage
  and 40/40 inventory checks. The targeted adapter, schema/identity,
  field-ledger, inventory, and coverage suite passes 45 tests; the US generator
  check and targeted Ruff checks are also clean.
- Replaced the vacuous passive wrapper preservation proxy with current-main QBI
  reconciliation on both the baseline and passive-staged paths. Seed 13
  realizes exactly two nonzero rows (positions 25 and 36) while all 15 incumbent
  QBI leaves retain identical dtype and bytes.
- Strengthened the production multispine regression with a $2 million
  partnership-income fixture. Production seed 0 realizes exactly one nonzero
  row (position 5, amount `158609.82342210703`) and every incumbent QBI leaf is
  byte-identical to the no-passive current-main pipeline. The independent
  full-length RNG-family array test remains in place; all three focused tests
  pass and targeted Ruff is clean.
- Made `us_release_input_coverage_gate` require every hard-required manifest
  column to exist in `engine.variables()`, independently of frame presence and
  signal. Failures name the version read from `engine_abi.lock.json` and report
  `required_missing_from_engine_registry` in gate details.
- Added the two explicit failure-mode regressions
  `test_required_all_zero_column_fails_when_registry_contains_it` and
  `test_required_column_absent_from_locked_registry_fails_even_with_signal`.
  The former proves default-only data stays red even with registry presence;
  the latter proves nonzero data stays red when the locked variable is absent.
- Documented in the generated release manifest, passive assumptions, repeal
  benchmark resource, and US fact-to-target guide that the passive column is
  engine-inert under PolicyEngine-US 1.764.6 and the hard release gate remains
  red until the pin advances past 1.764.6 to a release containing #9306.
- Regenerated the manifest and restricted assumptions, then propagated their
  identity through the US bundle (spec SHA-256
  `00b4c73b0ff2e29abf00b9f6f8112c9b87e937d323fb5c973bb5cd0652a95931`)
  and F0 report. The targeted gate/resource/inventory/adapter suite passes 153
  tests in 8m34s; the generator check and targeted Ruff are clean.
- The 2026-08-21 restricted replay check called the production calibration
  solver and therefore did not constitute an independent solve. Its former
  certification claim is superseded by the 2026-08-22 B3 entry below. The
  ordinary assumptions validator change and its historical test results remain
  valid branch history.
- The implementation and verification bullets below describe the historical
  pre-review checkpoint at `a4d93f78`; the verdict supersedes its former
  completion claim.

- Read `CLAUDE.md` and established the PR-CI/certification and journal
  contracts.
- Confirmed the primary checkout has unrelated untracked paths and left it
  untouched.
- Created the requested isolated worktree at
  `.claude/worktrees/passive-722` from current `origin/main`.
- Attempted the GitNexus exploration workflow; no GitNexus MCP resources or
  tools are configured, so source and Git-history inspection are the fallback.
- Confirmed the old QBI v3 stack is opt-in, diverges before the repository
  rename, and does not expose its latent entity form on the output frame.
- Selected a sibling passive assignment stage before current QBI
  reconciliation. This preserves the archived 15-leaf QBI contract and all
  prior QBI random streams while still accepting a latent form for routing
  when one becomes available.
- Reproduced the six SCF Schedule-E-band cells from the local 2022 public
  extract. Presence cells all clear effective n=30; the three middle-band
  conditional-share cells fall back to the pooled holder sample.
- Confirmed the restricted PUF replay artifact is present and pinned its
  size, digest, row counts, weighted positive pass-through aggregate, and
  provisional Form 8960 midpoint calibration target.
- Added the reproducible provisional SCF evidence resource. Holding
  prevalence rises from 3.35% in the nonpositive Schedule-E band to 38.98%
  above $1 million; conditional-share cells in the three thin middle bands
  use the documented all-holder fallback.
- Added the version-1 sibling assignment and assumptions build. The persisted
  log-odds shift is `-1.157105426398319`; its expected aggregate is the
  $54.628492 billion midpoint and the seed-0 replay produces $55.021132
  billion (0.72% high, inside the 5% diagnostic tolerance).
- Verified the restricted replay (12 focused tests), deterministic assumptions
  regeneration, strict resource hashes, isolated PCG64 families, latent-form
  routing, and byte preservation of all 15 existing QBI leaves.
- Integrated the sibling between Schedule-D completion and QBI reconciliation
  in multispine and direct support builds, while preserving already-assigned
  rebuilt support. The 997-row remaining-stage audit and support/pool
  checkpoint identities now bind the evidence, assumptions, shift, and RNG.
- Added the new output to ownership, L0 export, and release-coverage contracts.
  The exact pending-engine exception is limited to the input arriving in
  PolicyEngine-US PR #9306 and becomes a no-op once that engine version lands.
- Added separate, diagnostics-only Form 8960 line 12 and line 4c aggregate
  rows. They cannot affect the release gate and retain row-local error
  containment under the currently locked pre-#9306 engine.
- Completed the exact final-tree workspace suite: 6,394 passed and 73 skipped
  in 42m45s. The restricted artifact replay separately passed all 12 tests.
- Completed repository-wide `ruff check .`, changed-file Ruff format checks,
  `git diff --check`, the spec-only/entrypoint guards, and a manual incumbent
  package-name sweep over the branch diff. The manual sweep is required because
  the ordinary tree guard deliberately excludes `.claude/` worktrees.
- Added and committed the towncrier fragment and wrote the completion report
  to `FINAL_REPORT.md`.

## 2026-08-22 — B1/B3 re-verdict closure

- **B1 maps to `bd537c42`.** A secrets-free CI test freezes SHA-pinned
  historical v1/v2/v3 simulators, reconciler, and assumptions from `cfbf6330`.
  Its three parameters each require positive passive realization, guard and
  byte-compare the complete literal 15-leaf QBI simulation surface with and
  without the passive stage, compare the routed self-employment result,
  preserve global NumPy RNG state, and compare all version-family draws. The
  targeted node passed all three parameters.
- **B3 maps to `92dd3568`.** The secrets-free CI test uses test-local JSON
  parsing, inverse-CDF integration, an odds transform, row-wise weighted
  aggregation, and bisection. It reproduces the hand-computable `log(3)` shift
  and `$187.50` target, then cross-checks the production solver; the targeted
  node passed once. These test-local routines import or call no production
  calibration helper.
- The B3 real replay remains env-gated. Ordinary CI collects it but explicitly
  skips when `POPULACE_PUF_2024_H5` is unset (`1 skipped` was exercised); a set
  but missing path raises `FileNotFoundError`. With the supplied restricted
  artifact it passed once: independently solved and committed shifts are both
  `-1.157105426398319`; the effective binary64 shift tolerance derived from
  the JSON's 128-iteration bracket is `2.220446049250313e-16`; the independently
  reconstructed seeded aggregate is `$55,021,131,518.061035` and its midpoint
  error is `0.7187449327011208%`.
- Touched-file Ruff check and format check are clean. No test run exited 137.
  The earlier full-suite counts above were not rerun during this high-load
  closure and must not be read as results for the new B1/B3 commits.

## Next

- The supervisor can push the local commits and let clean-runner CI arbitrate
  the suite. The PolicyEngine-US #9306 dependency and provisional calibration
  follow-ups remain unchanged; this task performs no promotion.
