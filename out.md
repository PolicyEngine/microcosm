# Gate-failed base-pool release lane: final report

Date: 2026-08-26

Branch: `release-from-gate-failed-pool`

Base: `origin/main` at `2263df36`

## Outcome

The legacy `--base-h5` release and preflight paths now fail closed when the H5
identifies as a US multispine pool. They require and authenticate the canonical
sibling pool manifest, bind it to the exact requested H5, and reject a red
terminal agreement battery by default. The new explicit opt-in is:

`--allow-gate-failed-base-pool`

The flag is valid only with `--base-h5` and only for an authenticated current
stacked pool with `status=gate_failed` and `simulation_ready=false`. It is
rejected for generic H5 inputs, green pools, and the exact-k manifest arm. A
release built with the flag carries the full authenticated red verdict so a
reader can see `battery: red, N failures` without fetching the pool.

Publication preflight authenticates the same pool and displays the red receipt
as prominent human-review evidence. The red battery alone does not change the
existing preflight PASS/AT_RISK/FAIL calculation or exit code. Publication
remains a separate human-gated operation.

No network access, pool build, release build, publication, push, battery
threshold change, or gate-logic change was performed.

## Diff summary and rationale

- `CLAUDE.md`: documents the authenticated `--base-h5` boundary, explicit red
  opt-in, verdict carriage, strict exact-k arm, and separate publication step.
- `PROGRESS.md`: maintains the required state/done/next journal from kickoff
  through the final verified handoff while preserving the prior lane history.
- `changelog.d/gate-failed-pool-release.fixed.md`: records the closed receipt
  bypass and explicit opt-in under the repository changelog convention.
- `packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py`: adds the
  shared pool classifier, exact-H5 binding check, release-specific authenticated
  loader, and normalized self-contained release receipt. The existing strict
  simulation-ready and scoring loader contracts were not weakened.
- `packages/microcosm-build/src/microcosm/build/us_runtime/release_gate_preflight.py`:
  authenticates identified base pools through the shared seam, records their
  receipt in `PreflightReport`, and renders authenticated red evidence before
  the ordinary check table without including it in exit-code calculation.
- `tools/build_us_fiscal_refresh_release.py`: adds
  `--allow-gate-failed-base-pool`, authenticates pool-like legacy base H5s before
  the generic loader can run, uses the authenticated frame/H5 identity, and
  passes the receipt into both generated manifests. The exact-k arm remains
  simulation-ready-only.
- `tools/preflight_us_release_gates.py`: adds the same explicit opt-in and an
  optional `--release-manifest`; validates the carried full verdict, requires
  its receipt to exactly equal the pool authenticated from `--base-h5`, emits a
  large red/human-review banner, and adds machine-readable carried evidence.
  Existing required `--base-h5` and `--selection-source-manifest` arguments and
  all other preflight behavior remain intact.
- `packages/microcosm-build/tests/test_us_multispine_pool_h5_io.py`: exercises
  real stamped H5 classification, strict/default refusal, explicit red
  acceptance, redundant-green refusal, receipt contents, identity binding, and
  malformed aggregate/nested verdict refusal. PyTables-dependent cases use the
  sibling-test `importorskip` idiom.
- `packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py`: covers the
  old bare-H5 bypass directly, red-manifest default refusal, flag routing,
  generic/non-pool refusal when the flag is present, both-manifest carriage,
  exact-k isolation, and AST-level ordinary base-pool wiring guards.
- `packages/microcosm-build/tests/test_us_release_gate_preflight.py`: covers
  bare/missing/red receipt authentication, prominent text and JSON carriage,
  nonblocking PASS/AT_RISK/FAIL exit semantics, required static inputs, and
  exact mismatch refusal across manifest/H5/publication/gates/full-verdict
  identities.
- `out.md`: this final implementation, audit, verification, and handoff report.

## Flag and manifest contract

The exact flag is `--allow-gate-failed-base-pool`. The release builder permits
it only on its legacy `--base-h5` arm. Publication preflight uses the same flag
to authenticate and inspect the same red pool; neither invocation is an
authorization to publish.

The build manifest stores the receipt at top-level `base_pool`. The release
manifest stores the same receipt at `build.base_pool`:

- `artifact_kind`
- `status`
- `simulation_ready`
- `manifest_sha256`
- `publication_run_id`
- `pool_h5_sha256`
- `pool_h5_size_bytes`
- `allow_gate_failed_base_pool`
- `agreement_gate_reference.battery_status`
- `agreement_gate_reference.passed`
- `agreement_gate_reference.gates_json_sha256`
- `agreement_gate_reference.failure_count`
- `agreement_gate_reference.failures`, a flattened list of `{gate, message}`
- `agreement_gate_reference.verdict`, the complete agreement-gate verdict

The receipt accepts only coherent pairs: simulation-ready/green/passed or
gate-failed/red/failed. Every nested gate must carry a boolean `passed` and a
coherent failure list; the nested aggregate must match the terminal status.
For a red pool the failure list must be nonempty. SHA-256, publication run ID,
H5 size, full verdict, and all other receipt fields are bound through exact
preflight equality.

The preflight report also exposes the authenticated receipt as `base_pool`.
When `--release-manifest` supplies a red carried receipt, JSON adds
`carried_base_pool_agreement_battery`, including `battery_status=red`, the
failure count/list, gates digest, full agreement reference,
`publication_decision=human_review_required`, and `affects_exit_code=false`.

## §4 consumer audit

The requested exhaustive source audit used
`simulation_ready|gate_failed|load_simulation_ready` across `tools/` and
`packages/`, followed by source tracing of every non-test match.

- `h5_io.py` owns the authentication boundary. The new release wrapper selects
  the unchanged strict loader by default and the existing private terminal
  loader only when its required explicit boolean is true. The public scoring
  and strict simulation-ready contracts remain unchanged.
- `tools/build_us_exact_k_ladder_release.py` remains deliberately strict: it
  uses the simulation-ready loader, validates config/release identities, checks
  `agreement_gate.passed`, and never forwards the new flag.
- `tools/score_us_release_head_to_head.py` retains its existing authenticated
  terminal-evidence exception. It requires a pool manifest and terminal gates;
  it cannot authorize a release or accept a naked pool H5.
- `tools/build_us_multispine_pool.py` and
  `us_runtime/multispine_pool.py` are producers of the status, readiness, H5
  stamp, manifest, diagnostics, and terminal verdict rather than downstream
  release consumers.
- `tools/_legacy/build_us_acs_multispine_base.py` writes its own pre-calibration
  `simulation_ready=false` state; it is not a current terminal-pool consumer.
- `tools/build_us_acs_local_release.py` uses its own calibrated-release
  readiness summary. A derivative keeps donor release/revision provenance but
  does not project the nested `build.base_pool` verdict into its own manifest.
- The microcosm-data release contract, loader, TRACE conversion, and publisher
  tolerate but otherwise ignore the additive `build.base_pool` object. The
  designated publication preflight is therefore the prominent human-facing
  red-verdict surface.

Report-only interactions left unchanged, as directed:

1. `tools/build_us_multispine_pool.py::_stacked_manifest_payload` still labels
   `calibration.consumer` as `k-ladder` and says
   `requires_manifest_simulation_ready=true`. That metadata is now incomplete
   for the explicit legacy base-H5 red-pool route, but it is informational and
   unenforced.
2. A red pool producer writes its H5, manifest, gates evidence, and failed
   Logbook row, then returns status 1. Existing `set -e` candidate chains stop
   there, so an operator must deliberately start the separate release command
   to use the new opt-in.
3. ACS-local derivative manifests retain fetchable donor provenance but do not
   self-contain the nested red receipt.
4. Generic data/release consumers accept the additive receipt but do not
   surface it. This is not a publication bypass because the publication tool's
   designated preflight now authenticates and displays it.

No audit-only consumer was modified.

## Judgment calls

- Pool detection uses either a canonical sibling manifest whose artifact kind
  is the pool kind or the H5's own artifact-metadata stamp. Either positive
  identity requires the sidecar; sidecar existence alone is never trusted.
- A dedicated release loader with a required
  `allow_terminal_gate_failure` argument keeps the strict and scoring APIs
  semantically stable.
- The opt-in is rejected for a green pool rather than silently accepted, so
  every recorded flag use has one unambiguous meaning.
- Preflight requires exact equality between the release-carried receipt and the
  freshly authenticated base receipt. This prevents displaying release A's red
  evidence while checking release B's pool.
- The earlier salvage branch was used as a reference and selected commits were
  replayed only after line-by-line review. The final tree corrected its
  manifest-only preflight expansion, loader naming/contract ambiguity,
  redundant-green behavior, receipt binding, nested-verdict coherence, and an
  inaccurate exact-k diagnostic. Full verification ran on the corrected tree.

## Verification evidence

All commands ran offline against the pre-built environment with
`uv run --no-sync` (and a task-local UV cache where required). Each pytest
shard ran in one independent process.

- `uv run --no-sync ruff check .`: PASS, `All checks passed!`
- `uv run --no-sync pytest -q packages/microcosm-build/tests`: PASS, 6,545
  passed, 45 skipped, 2,351 warnings in 3,326.74 seconds.
- `uv run --no-sync pytest -q packages/microcosm-calibrate/tests`: PASS, 203
  passed, 2 warnings in 11.39 seconds.
- `uv run --no-sync pytest -q packages/microcosm-data/tests`: PASS, 318 passed,
  2 skipped in 9.11 seconds.
- `uv run --no-sync pytest -q packages/microcosm-fit/tests`: PASS, 93 passed,
  1 warning in 30.27 seconds.
- `uv run --no-sync pytest -q packages/microcosm-frame/tests`: PASS, 295
  passed, 36 skipped, 1 warning in 89.30 seconds.
- Full aggregate: 7,454 passed, 83 skipped.
- `.venv/bin/python tools/ci_test_groups.py --verify`: PASS,
  `tracked_test_files=309`, `verification=ok`.
- Focused changed-file suites plus exact-k E2E/launcher and data
  contract/release/publish-guard regressions: PASS; the full shard runs above
  subsequently covered the same tests on the final code tree.
- `git diff --check origin/main...HEAD`: PASS.

The build-shard warnings are expected numerical, pandas chained-assignment,
PolicyEngine division, and intentionally fragmented-frame test warnings; no
warning is a test failure and none originates in the new receipt path.

## Commit inventory

Implementation and journal commits before this final report carrier:

1. `34375fe6` Document gate-failed release lane kickoff
2. `5be9e49a` Preserve prior progress journal history
3. `c5a24c0f` Record release containment design review
4. `72274910` feat: allow explicit gate-failed pool release inputs
5. `e3d847d6` feat: surface carried red pool verdict in preflight
6. `46811fdb` fix: preserve default exact-k receipt fixtures
7. `dd1ad19a` Add shared multispine base-pool authentication seam
8. `271ee1fe` Close legacy base-H5 pool receipt bypass
9. `92134322` Authenticate and surface gate-failed pools in preflight
10. `cd39f756` Document gate-failed base-pool release boundary
11. `9db4694a` Tighten the authenticated red-pool release opt-in
12. `72c4c7a1` Bind carried pool verdicts to preflight inputs
13. `59c5759d` Record completed release containment work
14. `af20d3f5` Lock ordinary base-pool release wiring in tests

Commits after the list carry this report, the final `PROGRESS.md` state, and
their handoff cleanup. A report cannot embed the hash of the commit containing
its own final bytes, so `git log --oneline origin/main..HEAD` is the
authoritative complete inventory. At final handoff the branch is 17 commits
ahead of `origin/main` with no uncommitted paths.
