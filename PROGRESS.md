# US current-money graph slice — build progress

Branch `us-current-money-bridge-20260905`, base `e3cd8f3a2aecd0826279b374ddacf1bf406d4028`.
Authority: `/Users/maxghenis/architecture-reviews/microcosm-20260904/us-current-money-bridge-approved-plan.md`
(root amendments take precedence over Fable's `us-current-money-consumer-bridge-plan.md`).

## State

Pure modules, source preparation and engine admission are implemented, tested and
committed. The graph module, inventory stage, consumers-resource re-pin and the
direct/cold/cache-only parity runner are next. No genuine graph run, no push.

## Done

- Fresh worktree from the explicit approved base after status/fetch/base inspection.
- Locked interpreter: I's `.venv` (CPython 3.14.4, policyengine-us 1.819.0,
  policyengine-core 3.31.0), driven with this worktree's six shard `src` roots
  prepended on `PYTHONPATH`, so every `microcosm.*` import resolves here.
- `asec_current_money_selection.py` — `BoundCurrentMoneyBody` and
  `SelectedCurrentMoney`, token-gated and not `RestatedAsecMoney` subclasses,
  with their own codec. 9 tests.
- `cps_carried_current.py` — the 24 corrected person leaves plus the SPM
  childcare leaf, amounts from the selected artifact only, routing codes from
  declared raw code columns, raw dollar columns refused. 12 tests including a
  legacy byte oracle on 2024 rows.
- `asec_prepared_source.py` — the ten-file source directory contract, single-pass
  preparation through the reviewed loaders, graph dtype promotion with recorded
  transitions, and a path-free receipt. 10 tests on invented checkpoints.
- `asec_engine_evaluation.py` + `asec_current_money_engine_defaults_v1.json` —
  pinned closures for the four admitted roots, empty defaults allowlist, blocked
  roots proven blocked, and a real PolicyEngine-US run. 9 tests.

## Next

1. `graph_asec_prepared.py`: codec loader, four kernels, graph builder, registry.
2. `graph_sources.py` codec registration; `graph_implementation.py` stage
   `asec_prepared_v1`; `graph_implementation_inventory.json` roster.
3. Consumers resource status + `RESOURCE_PINS` re-issue and the identity ripple.
4. Direct / cold / cache-only parity runner in separate processes.
5. Docs, changelog, CI inventory verify.

## Constraints in force

- No genuine microdata read, no real graph run, no push, no publication.
- Tiny invented fixtures only; private pins substituted in test scope only.
- Do not weaken any existing fail-closed contract.
