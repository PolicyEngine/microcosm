# Lane journal — native scale: transport ceiling and per-node retention

Branch `native-scale-transport`, worktree
`~/PolicyEngine/_worktrees/microcosm-native-scale`, stacked on
`native-verify-once` (PR #935). Draft PR only; never ready, never merged.

This file is a session-handoff journal in the sense CLAUDE.md means: accurate
when written, history afterwards. Check git/GitHub for current truth.

## State

**Started 2026-09-17.** Orientation complete; baseline run not yet launched.

## Charter, in one line

Lift the two hard limits the 2026-09-16 profiler found — the 64 MiB
whole-roster receipt transport (refuses above 6.1% of the US source) and
per-node full-population retention (~300 GiB at full source) — then prove a
build above the old ceiling at 1/10.

## Sites, line-cited in this tree (not the profiler's tree)

| what | where, at `5ff889814` |
|---|---|
| preparation payload limit | `survey_population_preparation.py:50` `MAX_PAYLOAD_BYTES`, `:104` `_encode`, `:108` `PAYLOAD_LIMIT` |
| the whole-roster document | `survey_population_preparation.py:1698-1699` (`"selection": _plan_document(plan)`, `"origins": origins`) |
| second origin budget | `survey_population_preparation.py:816-818` `_bounded_append` → `ORIGIN_LIMIT` |
| allocation payload limit | `graph_survey_population.py:62` `ALLOCATION_MAX_BYTES`, `:344-369` `_allocation_payload` |
| per-cell frame identity | `survey_population_preparation.py:601` `_frame_cell_encode`, the `_frame_identity` loop |
| the observer that retains 19 populations | `graph_atomic_survey_financial.py` `observe(...)`, called from `executor.py` `_observer_snapshot` |

## Done

- Read the profiler's §5, §9, §10, §11 and the verify-once report's measurement
  section. Baseline to beat: **2,010.07 CPU-s / 2,016.08 s wall / 13.31 GB peak
  RSS** for the 19-node financial graph at 1/1000 (verify-once file C).
- Confirmed no live microcosm build on the machine; 128 GiB / 18 cores.
- Located the 19-node harness:
  `~/PolicyEngine/_worktrees/microcosm-verify-once/.measure/harness19_verify_once.py`.

## Next

1. Baseline run of the 19-node harness at 1/1000 in a throwaway worktree at the
   base commit, with an RSS-over-time sampler and runner-vs-node-loop split.
2. Design note `docs/us-native-scale-transport.md`.
3. Implement, with refusal tests, identity-equality proofs, replay proof.
4. 1/10 run above the old ceiling.
5. CI-shaped battery, draft PR.

## Session log, 2026-09-17

### Baseline (requirement 1) — done, and it settles the profiler's open question

One uncapped-by-node run of the 19-node financial graph at 1/1000, on the
branch point `5ff889814`, in the throwaway worktree
`~/PolicyEngine/_worktrees/microcosm-native-scale-baseline`, pid 70342 / pgid
70342 (session leader), RLIMIT_CPU hard 9,060 s, harness soft 9,000 CPU-s /
12,000 wall-s / 48 GiB, launched only after `vm_stat` showed 69.9 GB available.
Status `COMPLETED_NINETEEN_NODE`.

| | |
|---|---|
| wall | 2,048.93 s |
| CPU | **2,028.69 s** |
| peak RSS | 13,157,728,256 B = 13.16 GB / 12.25 GiB |
| loadavg | start [6.68, 6.97, 6.40], end [10.75, 9.68, 8.66] |
| runner call | 2,045.91 s wall / 2,026.78 s CPU; 3.02 s of process wall sits outside it |
| financial node loop, 19 nodes | 277.36 s |
| prefix node loop, 9 nodes | 55.93 s |
| **runner outside both node loops** | **1,712.62 s = 83.7% of the runner call** |

The four ASEC predictor nodes are 267.99 s of the financial loop's 277.36 s;
the other fifteen are 9.37 s. The retention slope is **not visible** in the RSS
trace at this fraction: 19 snapshots at the measured 9.6–9.8 B/cell over
1.70e6 cells is 0.31 GiB, inside the trace's own 1–2 GB sawtooth (peak 13.16 GB
at t=1303 s, 5.92 GB at exit). That corroborates the report's §9 rather than
adding to it.

### Done since

- Vectorised per-column frame seal in `survey_population_preparation`, proved
  byte-for-byte against the predecessor traversal oracle that
  `test_us_survey_frame_identity_encoding.py` already carried (160 tests; the
  file's own runtime fell 7.16 s → 0.67 s).
- Segmented, content-addressed transport for all four whole-roster receipts.
  `docs/us-native-scale-transport.md` has the design, the five measured ceilings
  in binding order, and the two refusals that narrow.
- 1,013 line citations across seven mechanisms read and adversarially verified.

### Next

1. After-run at 1/1000 on this head — **queued behind a memory gate** (another
   lane's `smoke.py`, pid 68076, holds 52.7 GB; the waiter polls every 60 s).
2. 1/10 run above the old ceiling.
3. Replay proof, pins, CI-shaped battery, draft PR.

### 2026-09-17, later

**Landed:** the segmented transport for all four whole-roster receipts, the
vectorised per-column frame seal, the hardening of the spill path, the ceiling
receipt, the seal rate receipt, the row-count ceiling census, and draft PR
[#945](https://github.com/PolicyEngine/microcosm/pull/945) against
`native-verify-once`.

**Measured:**

| | |
|---|---|
| ceiling, full source | single `_encode` **REFUSES** `PAYLOAD_LIMIT`; the segmented transport carries 1,099,892,722 B in 17 segments, digest equal to the stream digest |
| ceiling, 1/10 | single `_encode` **REFUSES**; transport carries 109,804,304 B in 2 segments |
| seal, blended on the frame's own dtype census | 0.5195 → 0.0677 µs/cell, **7.68×**, bytes asserted equal per column first |
| pins | **none move**: every inventory contract matches the tree, all ten stage manifests build |
| graph suite | 778 passed, 1 skipped |
| touched `us_runtime` tests | 359 passed |
| frame identity file | 187 passed (was 160), 15.62 s |

**Zero `packages/microcosm-graph` hunks**, so nothing in this PR is main-only.

**Not implemented, by decision and with the mechanism recorded:** the memory half
of the retention. `store.py` refuses a store-backed population snapshot three
ways (masked-storage zeroing at `:456`, `MultiIndex` at `:619`, `CategoricalDtype`
at `:486-487`), object identity is pinned at
`graph_atomic_survey_financial.py:1897`, and `same_replayed_population` needs the
whole object while the replay cannot exist until `run_graph` has returned. So
nineteen detached populations must live in RAM or on disk between observation and
comparison unless the object comparison becomes a content seal — a
verification-contract decision, put to Max as a question.
`docs/us-native-scale-transport.md` §4 specifies the executor mode item by item.

### Next

1. The 1/10 graph run (worktree `microcosm-native-scale-tenth`, launcher waits
   for 70 GB), after the 1/1000 after-run's required replay finishes.
2. Fold the after-run and 1/10 numbers into the report and the PR body.
