# US current-money graph slice — build progress

Branch `us-current-money-bridge-20260905`, base `e3cd8f3a2aecd0826279b374ddacf1bf406d4028`.
Authority: `/Users/maxghenis/architecture-reviews/microcosm-20260904/us-current-money-bridge-approved-plan.md`
(root amendments take precedence over Fable's `us-current-money-consumer-bridge-plan.md`).

## State

Resumed the stopped author's existing branch on 2026-09-05 at `025c0b61`.
Inspected all branch commits and the preserved graph implementation/source-codec
diff before editing. The original dirty files are retained. Completing the graph,
inventory, truthful consumer resource and tiny direct/cold/cache-only verification.
The real tiny direct/cold/cache-only parity check now passes; remaining negative
test verification and frozen provenance are in progress. No genuine graph run,
no push. This replacement journal stays local to the author
branch; root integration must omit it, as instructed.

Read the now-available Fable final APPROVE for S exact revision
`553084def66b5abd315e0e25bae9cd5e986d4867`. Locally composed its reviewed commits
as `dba326b0` and `907d7bf0`, after verifying the review and actual commit messages.
Root integration remains separate. Wiring S before tax reconstruction, binding
its separate receipt to unchanged T1/money, with housing authenticated on T1.

## Done

- Four graph kernels, codec registration, strict implementation scope (78 modules,
  six resources), and separately pinned current graph consumer status are complete.
  Original money consumer declarations/pins remain frozen to preserve T1 authority.
  The exact registry/resource/source suite passed 130 tests plus two focused
  final checks. Actual direct/cold/fresh-cache parity passed with zero warm
  kernel calls; remaining negative graph test validation is finishing.
- Independently compared full canonical legacy/T1 identity payloads and actual
  money-resource execution identity to base: all equal, with all 25 closure
  code/resource files matching both Git blobs. Receipt:
  `out/current-money-bridge/t1-closure.json`.
- Pinned complete installed PE-US/Core source and baseline parameter inventories;
  engine identity now verifies these before cache lookup and refuses shadow
  package/module origins. The four exact formula closures and empty defaults
  remain unchanged. Engine development checks: 12 passed, followed by 4 affected
  runtime-origin/real-engine checks, zero skips. Logs retained under
  `out/current-money-bridge/engine-*.log`.
- Resumed-source composition: separate reviewed S now attaches before actual
  tax-unit construction; housing authenticates the original T1. Prepared receipt
  binds S and T1 separately. Twelve targeted invented prepared-source tests pass,
  including original-NaN, original-source and money-byte preservation. Evidence:
  `/tmp/current-money-prepared-s-dev1.log` and `.xml` (26.758s JUnit).
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
6. Freeze and verify actual commit message; record loaded source hashes, exact
   tests and pending S composition in the author report. The requested external
   report directory is outside the session's writable roots; attempt export and
   retain a complete local output if the sandbox refuses it.

## Constraints in force

- No genuine microdata read, no real graph run, no push, no publication.
- Tiny invented fixtures only; private pins substituted in test scope only.
- Do not weaken any existing fail-closed contract.
