# Native attribution preflight: unavailable at the requested main base

Audit date: 2026-09-19. This is a code and existing-evidence audit, not a new
measurement. No native run, dataset build, calibration, or publication was
performed for this audit.

The requested 19-node cold run and required replay cannot execute against
`main` at `16c8e78d2f60d629da5d70294f643bce5c4597e2`: that tree does not contain
the native runner or its source-authentication modules. This is an absent-code
finding, not a timed refusal, memory-gate failure, or zero-cost result. Importing
the missing modules from a sibling checkout would produce a mixed-tree
measurement and would not reproduce the requested main baseline.

## Heads and absent modules

The lane's starting commit is `16c8e78d2`. At this audit its head was
`f00743c9a` (`docs: start parallel executor progress journal`); graph runtime
code was unchanged from the starting commit. The supplied transport checkout,
read without modification, was at `25d66f3c5` (`Record the 1/15 launch and the
prediction before its outcome`). Its historical measurements identify their
own earlier heads, separately below.

The following command inspected the committed base, independently of editable
installs, ignored files, or the active Python environment:

```sh
git ls-tree -r --name-only 16c8e78d2 packages/microcosm-build/src/microcosm/build/us_runtime
```

None of these files is in that inventory:

| File under `microcosm.build.us_runtime` | Relationship to the supplied harness |
|---|---|
| `graph_atomic_survey_financial.py` | Direct import; 19-node runner |
| `survey_population_preparation.py` | Direct import; source authentication and preparation |
| `survey_atomic_geography.py` | Direct import; independent geography reconstruction |
| `acs_housing_universe_source.py` | Direct import; ACS source custody |
| `acs_person_coverage_authentication.py` | Direct import; ACS coverage authentication |
| `acs_person_coverage_columns.py` | Direct import; literal ACS CSV reader |
| `asec_2024_native_population.py` | Direct import; ASEC native population |
| `asec_coverage_authentication.py` | Direct import; ASEC coverage authentication |
| `asec_current_money_source.py` | Direct import; ASEC money-source authority |
| `graph_atomic_survey_population.py` | Financial runner's prefix runner |
| `current_survey_predictors.py` | Financial predictor qualification |

The first nine imports are explicit in the
[transport harness at lines 218–228][harness-imports]. Its subsequent stray
module check requires every imported `microcosm` module to come from the
measured tree, so borrowing an installed copy is also contrary to the
harness's own measurement boundary.

## What #938 changed

The merge commit for [PR #938][pr938] is `8c44daa52`, with parent-two tip
`efe9cb6b5`. Reviewing the merge diff establishes that this was the graph-side
verify-once change, not the US native owner implementation.

- [`executor._object_stream`][object-stream] builds the same numeric/boolean
  context-digest byte stream using NumPy rather than a Python scalar loop.
  `_update_series` retains the old path for unsupported dtypes. This is a
  byte-encoding optimization, not a cache of live mutable frame seals.
- [`_SourceIdentities`][source-identities] caches source content keys for one
  `run_graph` call. Its stat signature includes the directory roster, file
  types, five stat fields, and followed member-symlink identities. A result is
  retained only when the signatures before and after deriving it agree.
- [Run-start authentication][run-start] populates that cache before node keys
  are derived; [the cold-node check][node-source-check] reuses it while the
  signature matches. This removes repeated full source reads at those node
  boundaries.
- [Run-end authentication][run-end] bypasses the cache and re-derives every
  source before returning the manifest. [`_settle_failed_run`][settle-failure]
  performs the corresponding check on an exception. Changed or unverifiable
  sources cause eviction of the run's complete recorded write set. The
  documented residual remains termination without unwinding before either
  final check.
- The executor accepts the caller's live verification-epoch statistics and
  attaches them to the manifest. It does not create the native owner epoch.

The native wrappers are instead present in commit `3bff62f85` on the native
history: the two atomic runners enter
`survey_population_preparation.verification_epoch()`. Those US files are absent
at this lane's base. Therefore #938 did not remove the US runner's source
issuance, HDF5 reconstruction, exhaustive CSV validation, predictor
qualification, or independent post-run population reconstruction. It also did
not create a persisted source-authentication receipt that those owners accept
in place of reconstructing their authority.

## Bucket mechanisms and memoization classification

The table's CPU columns describe the requested new main measurement. `N/A`
means it could not run because the modules above are absent; it never means
zero. The classifications below are derived from reading the transport tree's
code at `25d66f3c5`. They identify mathematical reuse opportunities separately
from what the current executor/store can actually change.

| Bucket | CPU-s cold at main | CPU-s replay at main | Classification and mechanism |
|---|---:|---:|---|
| ACS archive inventory, decompression, literal CSV validation | N/A | N/A | Potentially memoizable across runs for the same authenticated archive bytes and parser contract. Selected lineage additionally depends on the selected roster. `_inventory` exhausts archive members, hashes bytes, checks headers/widths/limits, and selects household/person lineage; the literal projection reader then scans again. Current owner issuance always performs these scans. [Source][acs-inventory] |
| ASEC private capture and CSV fencing, repeated by catalogue and native issuance | N/A | N/A | Repeated parsing of identical authenticated bytes is potentially memoizable within a run and across runs. Current-path custody is a separate obligation: `_capture` checks no-follow regular-file access, exact size, record/token bounds, complete digest, and stable descriptor identity while copying. A receipt cannot replace checking that the newly supplied file still matches. [Source][asec-capture] |
| ASEC HDF5 decoding and source reconstruction | N/A | N/A | Potentially memoizable across runs after authenticating the same complete file bytes and implementation contract. `_load` stages parent and attachment, reconstructs checkpoints, validates attachment/sidecars, derives scope, and issues current-money authority. It accepts no store receipt that skips this path. [Source][money-load] |
| Borrowing an already issued preparation/native capsule | N/A | N/A | Memoizable across nodes within a native verification epoch, and already memoized there. The signature binds source stats, owner/payload identities, frame-buffer witnesses and plan structure; authority, attached evidence, and producer encoding are still checked on every borrow, and final validation remains. [Source][native-memo] |
| Live frame seals and mutation checks | N/A | N/A | Irreducible at the current validation boundary. `_frame_signature` reads current metadata, cells, indexes and weights. Distinct node populations have different bytes; reusing an earlier live-object seal would miss changed cells. Repeated string token encoding within one column is already bounded and cached. [Source][money-seal] |
| Selection, normalized preparation, origins and context | N/A | N/A | Deterministic results are potentially memoizable across runs only under complete source, fraction, seed and producer identities. Results differ when those inputs differ. Current preparation reconstructs before comparing optional candidate bytes, and the issued capsule is process-owner-bound rather than a portable certificate. [Sources][preparation-issue], [capsule-check] |
| Predictor qualification and geography/population reconstruction | N/A | N/A | Results are specific to the selected populations and configuration, so distinct outputs are irreducible per node. Repeating an identical complete input derivation could in principle be memoized, but the current runner performs it independently before/after executor entry and the store has no interception point. [Source][runner-prefix] |
| Graph store payload integrity validation and decoding | N/A | N/A | Derived decoding can in principle be memoized against freshly verified bytes, with bounded memory and detached mutable results. Current disk-integrity verification is required on each access: metadata, payload size/SHA, exact roster and symlinks are checked. A stat-only validation cache weakens the existing refusal timing. [Source][store-verification] |

The classifications are not a claim that every opportunity can be implemented
in this lane. The material scope constraint is visible in the call graph:

1. The [financial runner][runner-prefix] executes its prefix, borrows the
   issued preparation, qualifies predictors, and reconstructs geography.
2. Only later does it call [`run_graph`][runner-executor]. By then the
   pre-executor cost has already been paid; caching inside that call cannot
   intercept it.
3. [`prepare_authenticated_survey_population`][preparation-issue] reconstructs
   source authority and calls `_validate` before checking `candidate ==
   payload`. A candidate receipt is evidence to compare, not permission to
   skip authentication.
4. [`AuthenticatedSurveyPopulationPreparation._checked`][capsule-check]
   requires the exact type, object identity, live weakref registration in
   `_ISSUED`, and unchanged issued payload. Persisting and reloading its bytes
   does not recreate that authority. Its checked view explicitly grants no
   authority of its own.
5. [Producer authentication][producer-check] compares both live producer
   identity and source bytes. Wrapping or replacing these functions is not a
   valid way to inject executor-side memoization or measure an unchanged
   producer. The supplied sampler deliberately avoids such wrappers.

No memoization patch to executor/store alone is justified as removing the
reported 84% on this evidence. Reusing native parsing/reconstruction needs an
explicit receipt-consumption boundary in the owning runtime, whose absence is
not repaired by naming a cache in the store. Such a boundary must preserve
source-change, producer-change, malformed-input and live-mutation refusals.

For the existing graph store, a successful first access does not authorize
skipping validation of later disk bytes. [`_verified_meta`][store-verification]
hashes every payload on every load; [the source-key optimization][source-identities]
has a different, explicitly documented contract that permits certain source
changes to be detected at run end. Extending that source contract to store
corruption without a contract change would be incorrect. A cache for decoded
results after current-byte verification may save decoding work, but retains
the read/hash cost and has not been attributed as a significant native bucket.

## Historical measurements: what they do and do not establish

The supplied [transport report][transport-report] and its two committed JSON
records were read directly. These numbers are historical evidence, not results
from this lane:

| Quantity | Baseline cold, `5ff889814` | After cold, `5307249b3` | After required replay, `5307249b3` |
|---|---:|---:|---:|
| Runner CPU seconds | 2,026.775 | 1,907.580 | 1,664.451 |
| Runner wall seconds | 2,045.909 | 1,975.986 | 1,699.618 |
| Financial node-loop wall seconds | 277.363 | 238.331 | Not separately recorded in the cited replay summary |
| Prefix node-loop wall seconds | 55.926 | 45.489 | Not separately recorded in the cited replay summary |
| Outside both loops, wall seconds | 1,712.620 | 1,692.166 | Not separately recorded in the cited replay summary |

The historical baseline process used 2,028.692 CPU-s; that includes process work
outside the runner. The familiar 333.29 seconds for both node loops and 1,712.62
seconds outside them are **wall** durations. Subtracting those wall durations
from CPU totals does not produce CPU attribution.

Both sampler records use a 0.25-second interval. The after-record has one
cumulative `sampled_cpu_ledger` spanning cold and required replay, with only
the top 60 chains retained. It cannot be split into cold and replay buckets.
The report explicitly acknowledges this in section 2a. Subtracting the cold
baseline ledger would also mix different code heads. Dividing the after-ledger
in half would assume the answer. Neither operation is a valid attribution.

The required replay's historical summary records all 19 nodes as store hits,
the same manifest key, and all 5,869 store objects unchanged. Its separate
`content_addressed_identical: false` projection is explained by the report's
`MappingProxyType`/`json.dumps(default=str)` instrumentation bug; it must not be
silently reported as a new successful canonical-byte check. The historical
manifest-key and store-byte checks themselves are recorded as successful.

Fresh phase-separated sampling remains necessary after an authorized native
runtime head is available. At this main base, no cold/replay bucket CPU totals,
native population weights, or native concurrency performance conclusions have
been established.

[harness-imports]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/experiments/native-scale-transport/harness19_after_with_required_replay.py#L218-L242
[pr938]: https://github.com/PolicyEngine/microcosm/pull/938
[object-stream]: https://github.com/PolicyEngine/microcosm/blob/16c8e78d2/packages/microcosm-graph/src/microcosm/graph/executor.py#L446-L538
[source-identities]: https://github.com/PolicyEngine/microcosm/blob/16c8e78d2/packages/microcosm-graph/src/microcosm/graph/executor.py#L2124-L2199
[run-start]: https://github.com/PolicyEngine/microcosm/blob/16c8e78d2/packages/microcosm-graph/src/microcosm/graph/executor.py#L2655-L2665
[node-source-check]: https://github.com/PolicyEngine/microcosm/blob/16c8e78d2/packages/microcosm-graph/src/microcosm/graph/executor.py#L2832-L2849
[run-end]: https://github.com/PolicyEngine/microcosm/blob/16c8e78d2/packages/microcosm-graph/src/microcosm/graph/executor.py#L3017-L3074
[settle-failure]: https://github.com/PolicyEngine/microcosm/blob/16c8e78d2/packages/microcosm-graph/src/microcosm/graph/executor.py#L2468-L2530
[acs-inventory]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/packages/microcosm-build/src/microcosm/build/us_runtime/acs_person_coverage_authentication.py#L426-L510
[asec-capture]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/packages/microcosm-build/src/microcosm/build/us_runtime/asec_coverage_authentication.py#L309-L345
[money-load]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/packages/microcosm-build/src/microcosm/build/us_runtime/asec_current_money_source.py#L534-L611
[native-memo]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/packages/microcosm-build/src/microcosm/build/us_runtime/survey_population_preparation.py#L1448-L1516
[money-seal]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/packages/microcosm-build/src/microcosm/build/us_runtime/asec_current_money_source.py#L98-L166
[preparation-issue]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/packages/microcosm-build/src/microcosm/build/us_runtime/survey_population_preparation.py#L1893-L2079
[capsule-check]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/packages/microcosm-build/src/microcosm/build/us_runtime/survey_population_preparation.py#L1806-L1870
[runner-prefix]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/packages/microcosm-build/src/microcosm/build/us_runtime/graph_atomic_survey_financial.py#L1350-L1409
[runner-executor]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/packages/microcosm-build/src/microcosm/build/us_runtime/graph_atomic_survey_financial.py#L1640-L1656
[producer-check]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/packages/microcosm-build/src/microcosm/build/us_runtime/survey_population_preparation.py#L708-L724
[store-verification]: https://github.com/PolicyEngine/microcosm/blob/16c8e78d2/packages/microcosm-graph/src/microcosm/graph/store.py#L228-L295
[transport-report]: https://github.com/PolicyEngine/microcosm/blob/25d66f3c5/experiments/native-scale-transport/out.md
