# US current-money graph slice — build progress

Branch `us-current-money-bridge-20260905`, base `e3cd8f3a2aecd0826279b374ddacf1bf406d4028`
(integration head named in the root approval). Authority:
`/Users/maxghenis/architecture-reviews/microcosm-20260904/us-current-money-bridge-approved-plan.md`
(root amendments take precedence over Fable's
`us-current-money-consumer-bridge-plan.md`).

## State

Setup complete; subsystem inspection running. No implementation committed yet.

## Done

- Fresh worktree from the explicit approved base after status/fetch/base inspection.
- Locked interpreter confirmed: I's `.venv` (CPython 3.14.4, policyengine-us 1.819.0,
  policyengine-core 3.31.0), driven with this worktree's six shard `src` roots
  prepended on `PYTHONPATH` so every `microcosm.*` import resolves here, not to I.

## Next

1. Finish reading real codec / context / closure / source integration.
2. Selection types + codec.
3. Corrected leaves module.
4. Source preparation module.
5. Graph module + codec registration + inventory stage.
6. Engine evaluation with pinned closures for the four admitted outputs.
7. Consumers resource status + pin re-issue, docs, changelog.
8. Tiny-fixture direct / cold / cache-only parity plus the real-PE test.

## Constraints in force

- No genuine microdata read, no real graph run, no push, no publication.
- Tiny invented fixtures only.
- Do not weaken any existing fail-closed contract.
