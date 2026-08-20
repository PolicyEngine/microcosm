# Progress: US reviewed-exclusion machinery

## State

Implementation is complete and final evidence refresh is in progress on
`battery-excl-machinery`,
which started from `origin/main` at `164027e2`. The empty schema-2 register,
exact criterion lifecycle machinery, authenticated source provenance,
terminal-manifest validation, CLI wiring, configuration-boundary hardening,
and adversarial tests are implemented and suite-green. The assigned scope
remained machinery only: no band, threshold, seed, fold, comparison algorithm,
or exclusion approval changed. Nothing was pushed.

## Done

- Confirmed the worktree started clean on the named branch.
- Read `CLAUDE.md`, the required adjudication report, and the machine-readable
  adjudication rows for the two dead QBI checks that the reviewed-exclusion
  finding explicitly classifies as `FIX-CANDIDATE`, not exclusion candidates.
- Started line-by-line inspection of every cited US mechanism and UK template
  module, plus read-only integration and verification audits.
- Attempted the required `uv sync --all-packages --extra us`. The default uv
  cache is not writable in the managed sandbox; a retry with a private cache
  reached resolution but could not download `hf-xet` because sandbox DNS is
  disabled. No dependency or lockfile changed.
- Attempted the GitNexus exploration workflow; no GitNexus MCP resources or
  tools are configured in this session, so discovery is using direct source
  and test call sites.
- Read every module and line range cited by the assigned adjudication rows,
  including the cited UK snapshot and the refactored equivalents on current
  `main`.
- Captured the existing 1% terminal receipt for both assigned checks: each is
  `dead_comparison`, with ASEC/ACS incidence 0/0 and nonzero rows 0/0.
- Confirmed a cold 1% build would violate the lane's 15 GiB RSS ceiling; the
  existing run records a primary-QRF peak of 84.73 GB (78.91 GiB). The after
  measurement therefore reran fresh terminal gates from its saved simulated
  frame.
- Recovered a runnable locked US environment by cloning the already-synced
  adjudication-worktree venv and rebinding its editable paths locally. The
  original exact sync command was still attempted first and remains
  network-blocked as recorded above.
- Baseline battery/manifest tests pass: 8 targeted tests.
- Added the declared, packaged schema-2 US by-origin exclusion resource with
  an intentionally empty `exclusions` object.
- Added a US-specific immutable receipt/policy loader: exact
  `leg#criterion` keys, exact five-field receipts, canonical date validation,
  duplicate/non-finite JSON rejection, canonical policy digest, and
  path-plus-SHA override authentication with partial-arming refusal.
- New schema/resource tests pass: 27 loader tests and 5 country-package tests;
  Ruff check is clean and both new Python files are Ruff-formatted.
- Hardened the exact-ID parser after adversarial review: malformed shorthand,
  noncanonical clone indices, invalid signed criteria, non-categorical joint
  criteria, and duplicate joint members now fail closed. An independent
  inventory check confirmed that all 417 unique criterion IDs emitted by the
  current canonical runtime surface remain accepted.
- Integrated exact criterion IDs into the existing comparator paths without
  changing their bands, thresholds, status records, or human failure text.
- Added in-force application and stale/dormant/expired/premature anti-rot
  classification. Only an exact in-force failing criterion is suppressed;
  structural invalidity and unevaluated criteria remain unsuppressed.
- Added an independent terminal-manifest validator that reconstructs criterion
  failures from comparison evidence, verifies the canonical policy digest and
  all five receipt fields, rejects inconsistent lifecycle buckets, and permits
  a passing `dead_comparison` only when its exact `#dead` receipt was applied.
- Added runtime tests covering the two assigned dead checks, all eligible
  comparator criteria, lifecycle boundaries, stale/dormant behavior, exact
  isolation, and post-evaluation receipt tampering.
- Added paired CLI options for an override path and raw-file SHA-256. The
  driver authenticates the complete policy before starting stacked work,
  rejects either half alone, and rejects any override on the legacy route.
- Added the authenticated override path to publication/checkpoint overwrite
  protection while deliberately keeping the policy out of simulation input
  pins and checkpoint identity. Regression tests cover both collision classes.
- Threaded the typed policy through cold and resumed stacked execution into
  the fresh terminal battery. It is deliberately excluded from input pins and
  simulation checkpoint identity, while the resulting report remains in the
  sealed terminal receipt and final manifest.
- Replayed the fresh terminal battery from the authenticated saved 1% simulated
  checkpoint (sample seed 578) at 1,544,634,368 bytes peak RSS. The manifest
  validator accepted the new committed empty-policy report.
- Recorded the honest per-check result: both assigned checks remain
  `dead_comparison` at ASEC/ACS incidence 0/0 and rows 0/0, with no exclusion
  applied. This is unchanged by design: the adjudication rejects receipts for
  these two `FIX-CANDIDATE` rows and assigns their per-target clone-role
  comparator correction to a separate fix lane.
- The first workspace-wide test run exposed the intended fail-closed runtime
  inventory: the new policy module was unclassified and the CLI graph count
  remained pinned at 64. Classified it explicitly as a reviewed
  non-population-treatment module, advanced the graph pin to 65, and confirmed
  the complete spine-blindness file passes.
- Confirmed the complete workspace suite passes at current HEAD with
  `.venv/bin/python -m pytest -q` (exit 0).
- Added a same-leg dual-failure regression proving one receipt suppresses only
  its exact incidence or quantile criterion while the other remains red.
- Reconfirmed the exact hardened tree with the complete workspace suite
  (`.venv/bin/python -m pytest -q`, exit 0), `.venv/bin/ruff check .`, Ruff
  formatting checks, `git diff --check`, and a direct schema-2/empty-register
  assertion.
- Attempted a wheel build from `packages/microcosm-build`; the sandbox could
  not resolve/download the isolated `hatchling` build dependency. The resource
  declaration and loading remain covered by the passing country-package and
  exclusion-loader tests, but an isolated wheel could not be produced here.
- Refreshed every mechanism citation in `_LANE-NOTES.md` and replaced the stale
  root report with this lane's `FINAL_REPORT.md`.
- Rewrote only this unpushed lane's local commits so the runtime-module
  classification lands with the register/loader and the 65-module import-graph
  pin lands with battery integration. The loader/country/spine-blindness suite
  is green at the former commit, the focused battery plus spine-blindness
  suites are green at the latter, and the override-driver tests are green at
  the driver commit.
- Independently re-audited checkpoint collision protection and closed a
  destructive sidecar gap: every deterministic stage H5, manifest, and
  operational-receipts path, plus the frame writer's deterministic temporary
  path, now rejects collision with an authenticated reviewed-exclusion
  override before cold build work.
- Hardened terminal battery reconstruction without changing any frozen
  doctrine: exact comparison/leg schemas, independent origin-scope receipts,
  incidence/count and ratio arithmetic, sealed conditional quantiles with
  recomputed QED, recomputed categorical TVD, and an exact tolerance receipt
  now fail closed. Adversarial tests cover each previously reproducible
  resealed-pass forgery. The complete stacked-spine test module and complete
  workspace suite pass at this commit boundary.
- Bound each canonical policy object to path-free provenance minted only by
  the same-byte hash-and-parse loader. The committed resource seals its raw
  package-byte digest; an override seals matching expected/raw digests and
  retains `override_supplied=true` even when its schema-2 content equals the
  committed register. Directly constructed policies remain available to the
  explicit noncanonical test seam but are refused at canonical battery, pool,
  and CLI boundaries.
- Added both-direction warm-resume coverage: a simulated checkpoint created
  with the committed empty policy resumes under an authenticated override,
  and one created under an override resumes under the committed policy. In
  both cases simulation stages stay cached and fresh terminal gates receive
  the current policy snapshot. The warm artifact asserts that the complete
  five-field receipts and source provenance survive identically in the final
  manifest, diagnostics, and immutable Logbook terminal receipt.
- Verified this authenticated-source step with 55 loader/package tests, 71
  focused exclusion/battery/driver tests, both warm-resume directions, the
  complete stacked-spine module, Ruff, `git diff --check`, and the complete
  workspace suite (`.venv/bin/python -m pytest -q`, exit 0).
- Replayed the current terminal battery and manifest reconstruction from the
  SHA/size/metadata-bound saved 1%/seed-578 simulated frame. The normal resume
  loader refused the old simulated/transferred checkpoints because their live
  late-primary-QRF worker binding has since changed, so no cached simulation
  was represented as resumable and no rebuild was attempted. The narrower
  read-only evidence replay authenticated the exact saved H5 and operational
  receipt sidecar before loading the simulation frame, then ran only the
  current battery and manifest verifier.
- The final replay stayed below the lane ceiling at 1,724,448,768 bytes peak
  RSS (about 1.61 GiB). It produced no durable artifact, supplied no Logbook
  predecessor, and did not touch `logbook-pending-chain.txt`.
- Final 1% evidence remains honest and unchanged for both assigned checks:
  `dead_comparison`, ASEC/ACS incidence 0/0, nonzero rows 0/0, no receipt
  applied. The current report seals schema 2, the committed raw-source digest,
  and empty resolved/applied/lifecycle buckets; `GateReport` accepted its
  reconstruction. The battery remains red with 127 failures by design.
- Closed the final source-evidence audit gap: path-free provenance now seals
  the exact UTF-8 register text as well as its expected/raw digest pair. The
  terminal validator hashes and strictly reparses that text and requires its
  schema-2 content to equal the reported policy snapshot. A new adversarial
  regression proves that mutating both digest fields together no longer
  passes.
- Verified the exact-byte provenance correction with the 55-test
  loader/country-package set, focused terminal-forgery and both-direction
  warm-resume tests, Ruff and touched-file formatting, `git diff --check`, and
  a second complete workspace suite (`.venv/bin/python -m pytest -q`, exit 0).

## Next

- Repeat the terminal-only 1% replay with the exact-byte provenance schema,
  refresh the final handoff report, then leave owner review and push external
  to this lane. The two measured QBI rows remain non-exclusion `FIX-CANDIDATE`
  work for a separate per-target clone-role comparator lane; this lane's
  register must remain empty.

## Historical note

The previous contents described the completed mortgage donor outlier-screen
lane. That merged-session journal was historical on this branch as of
2026-08-19; repository history retains it at parent commit `164027e2`.
