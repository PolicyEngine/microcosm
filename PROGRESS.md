# US current-money graph slice — build progress

Branch `us-current-money-bridge-20260905`, base `e3cd8f3a2aecd0826279b374ddacf1bf406d4028`.
Authority: `/Users/maxghenis/architecture-reviews/microcosm-20260904/us-current-money-bridge-approved-plan.md`
(root amendments take precedence over Fable's `us-current-money-consumer-bridge-plan.md`).

## State

Implementation complete and frozen for root/native/Fable review. Exact-code-head
`d1a3a89ab2976d55f05a6d6c8f1f2f45e7770561` passed 106 targeted tests with zero
failures, errors or skips (JUnit 132.240s). The final commit changes this journal
only; all tested source and test blobs remain identical. No genuine-data read or
run, no push, no root integration, no release certification. Root must omit this
replacement journal from integration.

S frozen revision `553084def66b5abd315e0e25bae9cd5e986d4867` was composed locally
as `dba326b0`/`907d7bf0` only after reading its final Fable APPROVE. S attaches
before actual tax construction and binds separately to unchanged T1/money;
housing authenticates original T1. No local S constructor hook remains pending.

## Done

- Frozen 106-test verification: `out/current-money-bridge/frozen-1/receipt.json`.
  All 265 parent-process and 206 modules in each direct/cache worker matched
  exact own-worktree Git blobs; all three processes used explicit six shards.
  Cache-only replay: all four nodes hit, zero kernel/source/engine calls.
- Final changed-file Ruff lint/format, tracked CI file-group verification and
  `git diff --check` passed. Development failures remain intact under
  `out/current-money-bridge/development-logs/`; original stopped-author logs in
  the architecture review directory remain untouched.
- Four graph kernels, codec registration, strict implementation scope (78 modules,
  six resources), and separately pinned current graph consumer status are complete.
  Original money consumer declarations/pins remain frozen to preserve T1 authority.
  The exact registry/resource/source suite passed 130 tests plus two focused
  final checks. Actual direct/cold/fresh-cache parity passed with zero warm
  kernel calls; all 18 graph tests also passed in the frozen verification.
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

- Root/native/Fable review of the frozen bridge, followed by root-owned integration
  (omit author journals; avoid applying S twice). No genuine source/S production
  or graph run is authorized or performed by this branch.
- Final author report and exact source/test receipts are under
  `out/current-money-bridge/`. Export to the requested architecture-review path
  is subject to this session's filesystem restriction; retain local output on refusal.
- Stop. No further implementation or broadened tests are pending locally.

## Constraints in force

- No genuine microdata read or genuine-data graph run, no push, no publication.
- Tiny invented fixtures only; private pins substituted in test scope only.
- Do not weaken any existing fail-closed contract.
