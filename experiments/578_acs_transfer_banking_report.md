# Microcosm #578: ACS-transfer per-target banking report

## Outcome

The ACS-transfer phase now banks every completed ordered QRF model target in an
identity-specific directory. A retry loads each valid target, restores the
exact raw-draw prefix and every availability pattern's advanced QRF state, and
continues through missing, stale, or corrupt targets. The phase-complete
`transferred` checkpoint and the always-fresh agreement gate are unchanged.

The implementation commit is `346bcee4` (`Add ACS transfer target banking`).
No network access, push, production-data build, configuration change, or root
journal edit was performed.

## Banking design

The transfer derives an ordered bank index from the normalized, bounded target
registry and enters the banked path only when a store is supplied
([acs_transfer.py](../packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py#L880-L923)).
Within each active family it builds the ordinary availability-pattern QRF
contexts, then processes model targets in chain order. For each target it:

1. supplies the current per-pattern predecessor states to the store;
2. loads and validates a banked target, or calls `fit_draw_next` for every
   pattern;
3. stores the raw float64 draws and before/after states;
4. feeds those raw draws, not snapped or finalized values, into the next
   target; and
5. writes a checkpoint only after all patterns for that target complete.

That loop and the post-load transition validation are in
[acs_transfer.py](../packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py#L1284-L1580).
The approach preserves the QRF chain contract that later targets condition on
the exact raw prior draws
([qrf.py](../packages/microcosm-fit/src/microcosm/fit/qrf.py#L1151-L1195)).

Each target artifact contains:

- a target descriptor: global index/total, entity, bounded family, full family
  and model-target order, model target, and exported leaves;
- the raw draw as lossless little-endian uint64 bits plus a SHA-256 digest;
- one record per availability pattern with the predecessor-state digest and
  full advanced `QRFChainState`; and
- schema, materializer, full identity, target metadata, row count, and a
  canonical metadata digest.

The serialization and atomic write are implemented in
[acs_transfer_bank.py](../packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py#L232-L329).
Writes use a unique same-directory temporary HDF5, close/flush and `fsync` it,
then `os.replace` the final target path. A missing final with an orphan
temporary, a truncated/unreadable HDF5, a digest change, an unexpected target
descriptor, a row-count change, a pattern-order change, or a non-continuing
QRF state fails closed to rebuild. Named receipts distinguish `missing`,
`resumed`, `identity_mismatch`, `invalid_rebuild`, and `rebuilt`
([acs_transfer_bank.py](../packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py#L102-L230),
[acs_transfer_bank.py](../packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py#L331-L450)).

The prompt's live receipt describes 114 ACS-transfer targets. The checked-out
registry currently declares 118 exported leaves and 117 model targets before
recipient-native filtering, across 33 bounded families. The difference is
intentional rather than a hard-coded-count deviation: the runtime derives the
bank order from the identity-bound registry and banks every active model
target. The joint immigration codec is one model target for two exported
leaves and is covered explicitly by the tests.

## Identity binding and phase boundaries

Both intra-phase banks are namespaced by the same pool base-identity digest:

```text
<checkpoint-root>/primary-qrf/<base-identity-sha256>/...
<checkpoint-root>/acs-transfer/<base-identity-sha256>/targets/NNN__target.h5
```

The directory selection is in
[build_us_multispine_pool.py](../tools/build_us_multispine_pool.py#L334-L349).
Every ACS artifact additionally embeds the `transferred` stage identity
([build_us_multispine_pool.py](../tools/build_us_multispine_pool.py#L1416-L1432)).
That identity carries the pool checkpoint schema/materializer ledger, period,
seed, PolicyEngine-US version, all verified input pins (the six CLI roles),
operator registries and order, household shares, primary target order, ACS
target families and fit settings, the complete take-up contract, and simulation
batch size
([build_us_multispine_pool.py](../tools/build_us_multispine_pool.py#L619-L671)).
The target-bank artifact has its own schema-v1 and materializer-v2 ledger as an
additional fail-closed layer
([acs_transfer_bank.py](../packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py#L25-L35)).

The bank receipt is published inside `impute.acs_qrf_transfer.target_bank`
([build_us_multispine_pool.py](../tools/build_us_multispine_pool.py#L1447-L1474)).
It is acceleration evidence, not a new stage boundary. The `transferred`
checkpoint is still emitted only after the whole impute operator, including
tail transfer and all ACS targets, returns successfully
([multispine_pool.py](../packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py#L2067-L2085)).
The agreement gate is still evaluated after simulation/resume, outside the
checkpointed operators
([multispine_pool.py](../packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py#L2089-L2146)).

## Tail-transfer decision

The capital-gains tail transfer was not banked. Unlike ACS QRF transfer, it is
a deterministic joint-vector selection/assignment and validation pass with no
per-target forest fits
([puf_capital_gains_tail.py](../packages/microcosm-build/src/microcosm/build/us_runtime/puf_capital_gains_tail.py#L490-L605)).

Foreground fixture benchmark, timing only
`transfer_puf_capital_gains_tail` after imports:

| Receipt | Value |
| --- | ---: |
| Recipient rows | 8 on each of person, household, tax_unit, spm_unit, family, marital_unit |
| Donor rows | 3 |
| First call | 0.0554770410 s |
| Next 20 calls, median | 0.0544171870 s |
| Next 20 calls, min / max | 0.0232492080 s / 0.1271914580 s |

At this scale, a whole-frame checkpoint read/write would be comparable to or
larger than the work avoided and would add another identity/serialization
surface. The locally available run-5/run-6 operations note reports roughly 30
minutes for “QRF phase re-transfer,” but does not isolate tail transfer; the
production tool likewise profiles primary-QRF targets but not tail or ACS.
Therefore that note was not treated as evidence that tail itself is expensive.

## Parallelism report

ACS work does not parallelize across targets today. The family loop is serial
and the banked model-target loop is serial
([acs_transfer.py](../packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py#L887-L925),
[acs_transfer.py](../packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py#L1434-L1521)).
The ordinary QRF implementation also fits chained targets serially and draws
them serially because each target depends on its predecessors
([qrf.py](../packages/microcosm-fit/src/microcosm/fit/qrf.py#L1087-L1098),
[qrf.py](../packages/microcosm-fit/src/microcosm/fit/qrf.py#L1523-L1534)).
Primary-QRF target subprocesses are likewise launched synchronously, one
target at a time, rather than through a process pool
([puf_qrf_chain.py](../packages/microcosm-build/src/microcosm/build/us_runtime/puf_qrf_chain.py#L180-L225)).

One target can use multiple cores:

- forest fitting defaults to `n_jobs=-1` when
  `POPULACE_FIT_N_JOBS` is unset, and that width is passed to the quantile
  forest
  ([qrf.py](../packages/microcosm-fit/src/microcosm/fit/qrf.py#L404-L427),
  [qrf.py](../packages/microcosm-fit/src/microcosm/fit/qrf.py#L488-L514));
- prediction divides rows into chunks and uses a `ThreadPoolExecutor`, with an
  unset worker override resolving to `os.cpu_count()`
  ([qrf.py](../packages/microcosm-fit/src/microcosm/fit/qrf.py#L318-L350),
  [qrf.py](../packages/microcosm-fit/src/microcosm/fit/qrf.py#L430-L449)).

Thus `cpu=32` would be visible to intra-target fit and draw workers when those
overrides are unset. Whether it would *materially* reduce the observed phase
wall time is not established: the outer pattern/target dependency chain is
serial, and this workspace contains neither a 16-vs-32 benchmark nor a CPU
utilization receipt. No parallelism configuration was changed.

## Test and quality receipts

All suites ran in the foreground with the requested Python and relative
`PYTHONPATH`.

| Suite | Receipt |
| --- | ---: |
| Affected ACS transfer + pool-tool suite | 129 passed, 1 warning in 181.83 s |
| #583 source-spine-blindness guard | 495 passed in 5.73 s |
| Pool input-consumer surface guard | 169 passed in 6.94 s |
| Full workspace (`packages`, 5,089 collected) | 5,032 passed, 57 skipped, 7 warnings in 1,130.83 s |

The affected cases include cold bank vs monolith; interrupt after a durable
target and byte-identical resumed frame checkpoint; identity-mismatch rebuild
with named receipt; truncated final and orphan temporary rebuilds; a mixed
banked/missing/banked hole; the joint immigration codec; bound production
path/stage identity and bank receipt wiring; unchanged transferred boundary;
and unchanged fresh agreement behavior
([test_us_acs_transfer.py](../packages/microcosm-build/tests/test_us_acs_transfer.py#L894-L1105),
[test_us_multispine_pool_tool.py](../packages/microcosm-build/tests/test_us_multispine_pool_tool.py#L503-L747)).

The warnings were non-failing numeric/runtime warnings already exercised by
the workspace (joblib core detection, intentional overflow/non-finite test
fixtures, PolicyEngine-US division, and PyTorch sparse warnings). Changed-file
Ruff check and Ruff format check passed before the implementation commit;
final `git diff --check`, clean-tree, and mergeability receipts are recorded by
the final handoff after this report commit.

## Deviations and limits

- Tail transfer is deliberately unbanked for the measured reason above.
- No core-count or worker configuration changed; the 32-CPU conclusion is a
  code-path report, not a performance claim.
- The target count is registry-derived rather than pinned to 114, so registry
  evolution and the two-leaf joint codec cannot silently desynchronize bank
  indexes.
- No production artifacts were available or authorized in this no-network
  worktree, so verification used real QRF fixture paths plus the complete
  sandbox-allowed workspace suite.
