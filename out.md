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
