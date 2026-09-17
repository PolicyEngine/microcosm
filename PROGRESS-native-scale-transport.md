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
