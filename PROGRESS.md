# Progress: round 11 checkpoint nullable booleans

## State

The Round 11 failure audit is complete on `tail-stratum-support-652`. The real
1% US build completed the full late producer DAG in memory and failed only
while serializing the durable stacked `transferred` checkpoint. The current
shared frame-checkpoint schema rejects pandas nullable `boolean`, and
`person.is_female` is simply the first of 39 such columns in table order.

## Done

- Confirmed the checkout is clean, on `tail-stratum-support-652`, and exactly
  at `cd4faa33` before changes.
- Confirmed that commit already merges the locally available `origin/main` at
  `d1714a7c`; no network operation was performed.
- Loaded the repository, PolicyEngine data, development-standard, and
  debugging guidance.
- Attempted the GitNexus debugging workflow. Repository parsing completed, but
  the managed filesystem denied its global registry write. Removed the partial
  local `.gitnexus` index and completed the audit from source, tests, and the
  supplied smoke-r8 artifacts instead.
- Located the exact failure path:
  `build_stacked_pool` emits `stage="transferred"` through
  `_PoolStageCheckpointStore.write`, which reaches `_series_spec` before the
  checkpoint destination is touched. The earlier 81,434,791-byte
  `assembled.checkpoint.h5` completed, while no partial transferred H5 or
  sidecars exist. The Logbook row correctly stops after `puf_passed` because
  the `transferred` phase mark follows the durable write.
- Audited every durable stacked checkpoint boundary and every extension dtype:
  `assembled` has 17 supported `StringDtype` columns and no nullable booleans;
  `transferred` has 39 complete `BooleanDtype` columns and 19 `StringDtype`
  columns; the stored `simulated` evaluation frame has the same 39 + 19.
  No `Int64`, `Float64`, categorical, or other extension dtype reaches these
  boundaries.
- Enumerated the 39 nullable booleans: 20 gap-fill registry targets, 17
  post-PUF registry targets, and the source-native `person.is_female` and
  `person.is_household_head`. The simulated stage's eleven seeded take-up
  outputs remain NumPy `bool`, so they do not expand this set.
- Confirmed why lossless null support is still mandatory: before peer transfer,
  source alignment creates declared absences on the opposite spine, including
  the eight QBI boolean outputs outside PUF detail. The durable transferred
  frame happens to be complete, but shared machinery must preserve these masks
  whenever another legitimate boundary retains them.
- Audited shared consumers: outer-stage runtime (including UK national stage
  checkpoints), US ASEC raw-stage checkpoints, PUF support equivalence/raw
  checkpoints, primary-QRF banks, and legacy and stacked pool stores all use
  this codec. UK rowwise publication and ACS per-target banks use separate HDF
  codecs. Existing sampled artifacts on those other paths carry only supported
  strings or NumPy dtypes.
- Established the compatibility constraint: retain the frozen artifact kind,
  HDF root, and dataset identifiers; emit the existing schema-v2 bytes for
  frames without the new encoding; accept legacy v2 on load; use a bumped
  schema only when nullable data is present; and bump the pool checkpoint
  envelope materializer so stale serializer bytes cannot resume silently while
  leaving the stacked producer identity and its 182 valid target banks intact.
- Added the Round 11 red-test matrix. It covers complete and missing nullable
  booleans in entity and link tables, explicit mask corruption, forged-v2
  metadata, deterministic rewrite, the actual 131-target canonical metric
  registry, the exact 39-column stacked boundary inventory, pool-store reloads,
  and pinned byte goldens for a generic schema-v2 frame and the UK outer-stage
  checkpoint. The pre-fix run fails only at the intended BooleanDtype refusal;
  the inventory and both unchanged-byte goldens already pass.
- Implemented conditional frame-checkpoint schema v3 for pandas nullable
  booleans. Complete columns write canonical NumPy-bool values without a mask;
  columns with declared absences write the same values with masked storage bits
  normalized to false plus an aligned uint8 0/1 null mask. Both reload as
  `BooleanDtype` with exact logical values and absences.
- Kept schema-v2 emission byte-identical whenever the new encoding is absent,
  while the loader accepts both v2 and v3 and fails closed on downgraded,
  mismatched, noncanonical, wrong-rank, wrong-dtype, nonbinary, missing, or
  unexpected nullable-boolean data and masks.
- Bumped the shared pool checkpoint envelope materializer from 5 to 6 while
  retaining stacked producer identity 10 and legacy materializer 3. A dedicated
  regression proves a v5 envelope is rejected without changing the stacked
  base/bank identity, preserving the 182 completed smoke-r8 target banks.
- Passed the implementation slice: 37 tests across the complete checkpoint
  codec, canonical registry and extension inventory, stacked identity/envelope
  seam, pool-store reload, and all UK stage-checkpoint tests.

## Next

- Run the requested focused, #583, full-workspace, lint, format, and golden
  proofs, then perform an independent review cycle.
