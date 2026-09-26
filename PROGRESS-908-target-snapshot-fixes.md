# microcosm#908 — closing the four reviewed findings on the snapshot slice

Lane journal for branch `calibration-target-snapshots-908-fixes-20260912`.
Journals are history, not state (see CLAUDE.md): check git/GitHub for current truth.

Historical note, 12 September 2026: this journal records the earlier repair
lane. The lane closed at `9a325c8b0`; root independently repaired the remaining
strict-null codec case in `b8380c475`, with 52 tests passing. Current scope and
remaining integration work are in [the maintained guide](docs/calibration-target-snapshots.md).
The earlier work-in-progress and not-pushed statements below are historical.

## State

Merged the completed slice (`bae1887ff`) onto `origin/main` `116d46ee9`, resolved
the four overlapping spec-engine identity pins by recomputing them on the merge
ref, and am now implementing the four reviewed findings. Not pushed, no PR.

## Review basis

`908-final-review-review-full-report.md` (head `bae1887ff`, merge base
`35fc76dd1`) with `probe.py` / `reproductions.json`. Four open findings:

1. **C1** — the aggregate-only claim is enforced by a denylist over arbitrary
   nested JSON, so record-level vectors ride through `context` and a
   mapping-valued `candidate_id`.
2. **A1** — nested metadata is copied only by `dict(...)`, so a sink mutating a
   delivered payload changes the caller's mapping and the next snapshot.
3. **A2** — `_write_chunk` creates the final `history/<n>.json` name before
   writing it, so a concurrent reader sees a partial chunk, and a failed write
   leaves an invalid immutable chunk behind.
4. **A3** — the public codec admits impossible metadata: `epoch > epochs`, a
   non-timestamp `created_at`, a non-string `candidate_id`, and a
   `best_retained` whose availability, epoch and loss contradict each other.

## Done

- Merged `bae1887ff` into this branch and recomputed every conflicting
  identity pin on the merge ref (table below).
- 14 red regressions in `packages/microcosm-calibrate/tests/test_target_snapshots.py`
  (9 of them failing on the reviewed head), then the implementation:
  - **C1** — a closed, typed, bounded aggregate metadata contract
    (`normalize_metadata`, `normalize_best_retained`, `METADATA_LOCATIONS`,
    `MAX_METADATA_*`) applied uniformly to `context`, `search`, `selection`
    and `best_retained`, with string-only identifiers checked rather than
    coerced. A list is not a scalar, so a record vector has no shape to ride
    in at any depth. The record-level key-name rule is kept on top, because a
    *scalar* `household_id` is still record-level identity.
  - **A1** — every metadata container in a delivered payload is freshly built
    from immutable scalars, pinned by a structural test asserting the payload
    shares no mutable object with the caller, the observer, the bound view or
    the next snapshot.
  - **A2** — history chunks are written and `fsync`-ed to a hidden temporary
    and then published atomically with `os.link`, which refuses rather than
    overwrites an existing immutable chunk. Failed writes clean up their
    temporary. Atomic `latest.json`, run ownership, duplicate refusal and
    bounded retention are unchanged.
  - **A3** — strict public-codec validation of `created_at` (timezone-aware
    ISO-8601), identifiers, `epoch <= epochs`, `sequence >= 1`,
    `non_finite_rows <= n_targets`, the closed `best_retained` triple's
    availability/epoch/loss consistency, and a `selected` snapshot whose epoch
    contradicts its own selection receipt — paired with normalization at the
    emitting edge so honest nonfinite values still serialize as explicit nulls
    and counts and the observer still cannot abort a run.
- `experiments/908_review_findings_recheck.py` replays the review's own four
  counterexamples against this branch; receipt in
  `experiments/908-review-findings-recheck.json`. All four report closed.

## Deliberately narrow choices

- `context` has **no in-tree producer**. Rather than invent a wider shape for a
  caller that does not exist, it takes the same flat scalar contract as the
  seams `solve.py` really emits. A caller that needs structure should add a
  typed seam and extend the contract, in the PR that adds the caller.
- The metadata bounds (32 entries, 64-character keys, 256-character strings)
  are the narrowest values that comfortably hold everything `solve.py` emits.

## Verification (isolated interpreter, no installs, threads=1)

`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-verified-lanes-20260910/.venv/bin/python -I -B -S`
with this worktree's shard `src` roots ahead of that venv's site-packages and
an import-path assertion, `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=1`.

| battery | result |
| --- | --- |
| `packages/microcosm-calibrate/tests` | 307 passed (12 s) |
| `packages/microcosm-fit/tests` | 124 passed (13 s) — main's QRF pins intact |
| `test_spec_engine_loader.py` + `test_us_multispine_pool_tool.py` | 197 passed (309 s) |
| `packages/microcosm-build/tests -k "inventory or coverage or seed"` | 448 passed, 2 skipped (343 s) |
| `packages/microcosm-graph/tests -k parity` | 25 passed (88 s) |
| `tools/spec_engine_coverage.py --check` | 42156/42156 fields, 41/41 inventory |
| `tools/ci_test_groups.py --verify` | `verification=ok`; the snapshot test file lands in fast `rest` / engine `us-am`, never `[defaulted]` |
| `ruff check .` | clean; `ruff format --check` clean on both touched files |

`test_target_snapshots.py` holds 52 tests, 16 of them new here and 2 existing
ones updated to the closed contract's refusal points. Replaying the whole file
against the reviewed head's `target_snapshots.py` (the only source that
differs) gives **12 failed, 40 passed** — 10 new red regressions plus the 2
updated tests. Every finding has at least one red regression:

- **C1** — `test_context_refuses_the_record_vectors_the_review_smuggled_through`,
  `test_metadata_refuses_arbitrary_nested_payloads`,
  `test_identifier_fields_must_be_strings`,
  `test_supported_scalar_metadata_survives_and_is_bounded`.
- **A1** — `test_metadata_refuses_arbitrary_nested_payloads` is the review's
  own aliasing counterexample; the detachment invariant itself is pinned by
  `test_a_delivered_snapshot_shares_no_mutable_object_with_its_caller` and
  `test_sink_mutation_cannot_reach_caller_metadata_or_the_next_snapshot`,
  which pass on both heads because the flat case was never the defect.
- **A2** — `test_a_history_chunk_is_published_only_after_its_bytes_are_complete`,
  `test_a_failed_chunk_write_leaves_no_partial_or_leftover_file`.
- **A3** — `test_codec_refuses_the_impossible_payloads_the_review_reproduced`,
  `test_non_finite_diagnostics_stay_null_statuses_rather_than_aborting`,
  `test_the_codec_is_stricter_than_the_emitting_edge_and_says_so`,
  `test_created_at_is_a_timezone_aware_timestamp`.

The remaining 6 new tests are preservation tests (the structured metadata
`solve.py` really emits, the epochs the solver really selects, duplicate chunk
refusal, and observer-off/observer-on weight parity on the L0, budget-search,
refit and proximal paths); they pass on both heads by design.

## Next

- Independent re-review by main before integration/PR updates.

## Identity pins recomputed on the merge ref

`microcosm.calibrate.solve` is attested (`_DIRECT_KERNEL_MODULES`); main's #912
edited the attested `microcosm.fit.qrf`, so both sides' pins were stale and the
merge conflicted on all four. Each was recomputed on the merged tree, after
first proving with the same interpreter that restoring **main's** `solve.py`
into the merged tree reproduces **main's** committed pins exactly — so the
drift is attributable to the slice's `solve.py` edit, not to an environment
leak.

| pin | main `116d46ee9` | slice `bae1887ff` | merged |
| --- | --- | --- | --- |
| `EXPECTED_HASHES["seed_protocol"]` | `553d5e0b…` | `f4dc507c…` | `d052fd87…` |
| `EXPECTED_HASHES["seed_map"]` | `20058e54…` | `32dea304…` | `d5a9694a…` |
| US resolved-spec `spec_sha256` | `1eeca53a…` | `35a3623b…` | `ff2c9703…` |
| minimal-spec loader golden | `b4659890…` | `a67bb78c…` | `8a240898…` |

`docs/evidence/spec-engine/us-f0-coverage.json` was regenerated with
`tools/spec_engine_coverage.py` (42156/42156 fields, 41/41 inventory checks).
The graph `calibrate.adam@1` parity pin merged cleanly and re-verified green
(25 parity tests). `fit.qrf` and `simulate` pins are untouched.
