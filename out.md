# scorerlegacy final report

## Outcome

The historical-incumbent defect is fixed and locally certified. The scorer now
normalizes every loaded historical H5 artifact against the same
period-sensitive PolicyEngine-US metadata index used by the strict release
builder gate. It proves the current formula dependency leaves are present on
their declared entities, drops formula-owned artifact columns, preserves all
Frame authority, and emits deterministic JSON and Markdown receipts.

The exact real acceptance command cleared the reported failure, loaded and
fully scored the live incumbent, and reached 19.20 GiB peak RSS without
crossing the 20 GiB guard. It then stopped before loading the candidate H5
because the authenticated candidate-25 manifest failed an unrelated existing
late-producer contract:

    ValueError: US stacked pool manifest
    /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.manifest.json
    late producer 'primary_puf_qrf': late primary-QRF worker binding changed.

Per the lane instruction, I did not bypass, weaken, or repair that separate
boundary. The scorer writes only after both sides finish, so neither
pool-vs-incumbent.json nor pool-vs-incumbent.md exists.

## Diff summary and rationale

- tools/score_us_release_head_to_head.py
  - Bumped the deterministic scorecard schema from 2 to 3.
  - Added one scorer-only historical-artifact normalization seam and applied it
    to entity H5, legacy flat H5, and authenticated pool-manifest loading.
  - Reused release._formula_owned_gate_adapter() and
    _engine_computed_columns(..., period=release.PERIOD), exactly matching the
    fresh-builder ownership authority without editing the release gate.
  - For every detected output, resolves the metadata index's authenticated
    dependency closure and refuses before dropping anything when any input leaf
    is absent from its declared entity table. The error deterministically maps
    each output to its sorted missing leaves.
  - Rebuilds a cleaned Frame only when needed and preserves weights, strata,
    mass log, and metadata.
  - Records normalization_receipts.historical_formula_owned_columns with count
    and sorted columns_by_entity for every artifact; clean artifacts explicitly
    seal count 0 and an empty entity map.
  - Renders the same receipt in Markdown and uses stable symbol-based mechanism
    citations for the scorer code.
- packages/microcosm-build/tests/test_us_release_head_to_head_scorer.py
  - Added a real entity-H5 fixture containing the formula-owned Marketplace
    alias and its interview leaf; it loads, drops, scores, receipts, and renders.
  - Added a fixture missing the leaf; loading refuses and names both the output
    and missing leaf.
  - Added a clean H5 fixture; scoring emits the exact empty JSON and Markdown
    receipt.
- changelog.d/us-historical-scorer-formula-columns.fixed.md
  - Added the repository-convention fix fragment.
- PROGRESS.md
  - Maintained the committed state/done/next lane journal from the first step.

Rationale: a historical release artifact records the input surface of the
engine that built it. When today's locked engine formula-owns one of those
columns, retaining the historical value would make one side of the comparison
override a current formula. Dropping it only after proving its current leaves
are present makes both artifacts flow through one engine version, while the
unchanged fresh-release builder continues to reject formula-owned source
columns.

## Real dropped-column receipt

The acceptance invocation exercised the same incumbent load seam before
scoring. After the later candidate-authentication failure, a fresh read-only
print of that exact loaded receipt produced:

~~~json
{
  "columns_by_entity": {
    "person": [
      "has_marketplace_health_coverage"
    ]
  },
  "count": 1
}
~~~

The required leaf
has_marketplace_health_coverage_at_interview was present on person. The
candidate receipt was not produced because authenticated manifest validation
failed before the candidate H5 could load. Focused coverage proves that a clean
loaded artifact emits {"count": 0, "columns_by_entity": {}}.

## Headline score result

Unavailable because the head-to-head did not complete and the scorer
deliberately writes no partial output. There is therefore no
pool-vs-incumbent.md from which to quote weighted loss or percent-within-10
figures. The incumbent alone completed all 32,842 registry targets, but its
in-memory partial payload was correctly not persisted after candidate
authentication failed.

## Verification evidence

- Required Ruff command: passed.
  - The managed sandbox rejected uv's default cache before execution.
  - Re-run offline with UV_CACHE_DIR set to a writable task-specific directory:
    uv run --no-sync ruff check . -> All checks passed.
- CI test inventory: tools/ci_test_groups.py --verify passed for 309 tracked
  test files; the scorer test remains in engine lane us-qs.
- Targeted scorer suite: 16 passed in 34.34 seconds.
- Full pytest, one separate process per shard:
  - build: 6,516 passed, 45 skipped, 0 failed in 3,461.09 seconds;
  - frame: 295 passed, 36 skipped, 0 failed in 97.72 seconds;
  - calibrate: 203 passed, 0 skipped, 0 failed in 11.61 seconds;
  - data: 318 passed, 2 skipped, 0 failed in 7.59 seconds;
  - fit: 93 passed, 0 skipped, 0 failed in 27.28 seconds.
  - aggregate: 7,425 passed, 83 skipped, 0 failed.
- git diff --check passed throughout.
- No network, push, pool/release build, publication, or validation bypass was
  used.

## Acceptance execution

Command run from this worktree:

    .venv/bin/python tools/score_us_release_head_to_head.py \
      --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5 \
      --candidate /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.manifest.json \
      --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl \
      --out-prefix /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/score-pool/pool-vs-incumbent

Observed milestones:

- Fiscal yardstick compiled at 2.45 GiB peak RSS.
- Live incumbent loaded at 3.70 GiB, clearing the original formula-owned-column
  export failure.
- All five incumbent chunks completed, each over 12 fixed household slices.
- Incumbent scoring state released at 19.20 GiB peak RSS.
- Candidate manifest authentication then failed before candidate H5 loading.
- Both requested scorecard output paths were confirmed absent.

Complete emitted traceback:

~~~text
Traceback (most recent call last):
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/tools/score_us_release_head_to_head.py", line 2300, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/tools/score_us_release_head_to_head.py", line 2268, in main
    payload = score_head_to_head(
        incumbent=args.incumbent,
    ...<8 lines>...
        candidate_manifest_sha256=args.candidate_manifest_sha256,
    )
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/tools/score_us_release_head_to_head.py", line 1797, in score_head_to_head
    loaded = load_artifact(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/tools/score_us_release_head_to_head.py", line 598, in load_artifact
    artifact = _load_pool_manifest(
        resolved,
        expected_manifest_sha256=expected_manifest_sha256,
    )
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/tools/score_us_release_head_to_head.py", line 505, in _load_pool_manifest
    frame, manifest, authenticated = load_authenticated_us_multispine_pool_for_scoring(
                                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        manifest_path,
        ^^^^^^^^^^^^^^
        expected_manifest_sha256=expected_manifest_sha256,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py", line 1281, in load_authenticated_us_multispine_pool_for_scoring
    return _load_us_multispine_pool(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        require_simulation_ready=False,
    )
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py", line 1297, in _load_us_multispine_pool
    manifest, authenticated_pool_h5 = _load_authenticated_us_multispine_pool_manifest(
                                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        manifest_path,
        ^^^^^^^^^^^^^^
        expected_manifest_sha256=expected_manifest_sha256,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        allow_terminal_gate_failure=not require_simulation_ready,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py", line 440, in _load_authenticated_us_multispine_pool_manifest
    _validate_stacked_late_dag_manifest_binding(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        manifest,
        ^^^^^^^^^
        manifest_path=manifest_path,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py", line 644, in _validate_stacked_late_dag_manifest_binding
    validate_stacked_late_producer_receipt(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        dag,
        ^^^^
        boundary=f"US stacked pool manifest {manifest_path}",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py", line 8051, in validate_stacked_late_producer_receipt
    previous_sha256 = _validate_late_execution_row(
        raw_row,
    ...<3 lines>...
        boundary=boundary,
    )
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py", line 7744, in _validate_late_execution_row
    _validate_late_available_input_receipt(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        receipt,
        ^^^^^^^^
    ...<3 lines>...
        boundary=f"{boundary} late producer {contract.name!r}",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py", line 5768, in _validate_late_available_input_receipt
    _validate_late_resource_binding(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        binding,
        ^^^^^^^^
    ...<3 lines>...
        boundary=boundary,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-scorer-legacy/packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py", line 5506, in _validate_late_resource_binding
    raise ValueError(f"{boundary}: late primary-QRF worker binding changed.")
ValueError: US stacked pool manifest /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.manifest.json late producer 'primary_puf_qrf': late primary-QRF worker binding changed.
~~~

## Judgment calls

- Applied automatic normalization to every scorer artifact route because all
  those Frames originate from persisted historical H5s; no skip flag exists.
- Used the gate's period-sensitive ownership classifier, not the runtime
  PolicyEngineUSEngine helper and not period-insensitive
  formula_owned_outputs().
- Required every dependency-closure leaf on its metadata-declared entity. The
  source closure is a conservative all-vintage static union, so a future dated
  formula can theoretically over-require an irrelevant leaf; fail-closed is the
  correct direction for this task, and the locked Marketplace alias is a
  direct one-edge closure with no ambiguity.
- Kept the fresh release builder's _assert_no_formula_owned_columns function
  and semantics byte-for-byte untouched.
- Bumped the output schema because the always-present receipt changes the
  deterministic JSON contract.
- Did not treat the later candidate authentication failure as authorization to
  weaken pool provenance. The next action belongs to candidate artifact
  production, not this scorer lane.

## Commits before the final report seal

- 61311bf5 Start historical scorer column lane
- 169a441a Normalize historical scorer formula columns
- 7a44561c Handle clean entities during scorer normalization
- 2a2df350 Record complete scorer verification
- 34dd3129 Keep scorer mechanism citations stable
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
