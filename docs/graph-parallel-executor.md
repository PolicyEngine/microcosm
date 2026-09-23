# Concurrent graph executor (amendment 26)

Status, 2026-09-23: implemented behind a flag and tested for byte identity on
the toy graphs and the native nineteen-node financial fixture. The default is
one worker, which is the sequential executor unchanged. No actual-data run
has been timed with more than one worker.

This note replaces the provisional design of 2026-09-19 (branch
`graph-parallel-executor`, `f55e594b1`, kept on origin as
`backup/native/graph-parallel-executor`). That note audited main at
`16c8e78d2` and proposed canonical admission, coordinator-only writes and a
memory reservation budget. The implementation keeps the first two and
departs from the note in four places, listed under
[Departures from the provisional design](#departures-from-the-provisional-design).

Code: [scheduling.py](../packages/microcosm-graph/src/microcosm/graph/scheduling.py)
(new) and [executor.py](../packages/microcosm-graph/src/microcosm/graph/executor.py)
(`run_graph`, `_execute_graph` and five helpers split out of the node loop).
`decl.py` and `kernel.py` are untouched, so the interface lock does not move.

## The switch

`run_graph(..., max_workers=N)`. `1` is the sequential executor. `None`, the
default, reads `MICROCOSM_GRAPH_MAX_WORKERS` and falls back to `1` when it is
unset or blank (`resolve_max_workers`). Anything that is not a positive
integer is refused before the run opens its store's write ledger; a
non-decimal environment value is refused too.

The environment variable exists for country runtimes. The US native runner
and its nested population graph call `run_graph` from modules whose bytes are
bound into the US stage implementation hashes, so adding a parameter to them
would move every US node key a second time. A caller sets the variable and
every nested `run_graph` in that process picks it up.

The worker count enters no node key, receipt, cache record or portable
manifest field.

## What runs where

| Step | Thread |
| --- | --- |
| Compile, every node key, typed contracts, required-replay preflight | Coordinator, before any worker exists |
| Blocker check, incumbent lookup, writer/numeric/tolerance scopes, artifact authentication, cache lookup | Coordinator |
| Context projection (`_project_context`) and its "before" digest | Coordinator, early or at the node's turn |
| `kernel.run(context)` and the "after" digest of that same context object (`invoke_kernel`) | A worker (named `microcosm-graph-worker-*`) when `max_workers > 1`; the coordinator when it is 1 |
| Result validation, patching the cumulative population, the population observer, every store write, the receipt | Coordinator, in `compiled.order`, through the same code as the sequential run |
| Failed-run settlement and the run-end source re-derivation | Coordinator, after every worker is joined |

When `max_workers > 1`, the coordinator runs no kernel itself: the node at
the canonical head is submitted to the pool like any other.

## Scheduling

`Speculation.fill(position)` scans `compiled.order` from the canonical head.
A node is ready when every entry of `compiled.predecessors[node]` has an
admitted receipt. A ready node is offered once. `prepare` then runs the
checks the node's turn would run, in the same order and against the same
admitted receipts, and declines when the node:

- has a kernel whose structural capability does not match its declaration;
- is blocked (amendment 23): its turn records it as unreached without a
  kernel call;
- is expected to restore its cached record under `resume="auto"`
  (`_cached_record_expected`); or
- raises anything during preparation. The turn repeats the same step and
  raises the same error at the sequential point.

Otherwise it projects the context, reads the node's declared byte inputs from
the store, takes the context digest and submits the kernel call. The scan
stops while `max_workers` calls are running or `max_workers` prepared calls
wait for their turns. Because the scan starts at the head, the head gets the
first free worker. While a turn waits for its own call, every worker that
frees up is offered the next ready node.

`resume="require"` never builds the pool: every node restores a record, so
there is nothing to run early.

## A node's turn

A turn that misses its cache claims the node's prepared call, if there is
one, and uses it in exactly two cases:

1. **Same incumbent object and equal scopes.** `patch` returns a new
   Population and never mutates its argument (`population.py`, `patch`
   docstring), and every other input to the projection (node key, source
   paths, byte-input keys) is fixed for the run. The early context is
   therefore the context the turn would project, and the turn reuses it.
2. **Equal context digest and equal scopes.** A sibling admitted since the
   early projection replaced the incumbent object. The turn projects its own
   context; if its digest equals the early one, and the tolerance and numeric
   scopes are equal, the precomputed outcome stands.

The scope comparison is needed because `_context_digest` covers tables,
weights, strata and byte inputs, but not the tolerance and numeric scopes a
kernel may read. In any other case the turn discards the precomputed call
and runs its own context on a worker. A discarded call that has already
started finishes on its thread and is never read.

Either way the outcome enters the existing code: a gate exception becomes a
failed verdict, any other exception refuses the run, a kernel-authored
execution record is rejected, a changed "after" digest is refused as input
mutation, and each declared source is re-checked, all at the node's own turn.

## Why the bytes are the sequential run's

The argument is induction over canonical admissions. Assume every node
before the head was admitted with the sequential run's result. Then:

- node keys, contracts and the required-replay preflight are computed before
  any worker exists and do not depend on the worker count;
- the head's admitted predecessors, and so its incumbent, scopes and byte
  inputs, are the sequential run's;
- the context the kernel received is either the object the turn would have
  projected (case 1) or has an equal digest and equal scopes (case 2). Its
  RNG was created from the node key at projection and had not been drawn
  from before the call;
- a kernel that honours its declared contract (output a function of the
  projected context, its declared sources and its node seed) returns the
  sequential result;
- validation, patching, observation, serialization and receipt creation are
  the unchanged sequential code, run on the same thread, in the same order.

So the head is admitted with the sequential run's result, and the next node
inherits the premise. A worker's result is read only at its node's turn, so
a failure is the sequential run's too: the same exception is raised at the
same node, and the store holds the same published prefix.

The argument needs three things the executor cannot check, all of which are
the caller's:

- each kernel honours its declared contract, and writes no declared source;
- each registered kernel instance, and any module state it shares with other
  kernels, is safe to call from a worker thread while others run;
- declared sources stay put for the run. The existing per-node check and
  run-end re-derivation still catch a change, now after the workers are
  joined.

Because of the second, the default stays one worker.

## Failures, joining and the write ledger

Workers never receive the store, never touch the cumulative populations and
never write receipts. Only coordinator admissions reach
`ContentStore.recording_writes`. `Speculation.close()` cancels every call not
yet started and joins every call that did. It runs on every exit from the
node loop: after the last admission, before the run-end source pass, and,
when the loop raises, in `run_graph`'s handler before `_settle_failed_run`.
No kernel is still reading a source while settlement re-derives it, and no
write can arrive after settlement copies the ledger.

## Observers

The private population observer runs on the coordinator, once per admitted
node, in `compiled.order`, before that node is persisted, exactly as it does
sequentially. No two callbacks overlap. Amendment 24's detached snapshot is
unchanged.

## Memory

Memory has a structural bound. There is no byte budget. Beyond the
sequential run, a parallel run holds at most:

- `max_workers` prepared contexts waiting for their turns, with their
  results once computed;
- discarded calls still running, which occupy pool threads, so there are at
  most `max_workers` of them, with their contexts;
- whatever the kernels allocate while they run concurrently. No worker count
  bounds this.

The third item dominates at scale. A native nineteen-node run at 1/15
peaked at 43.46 GiB with one worker (`overnight-20260923/native-line-state.md`,
section (c), from the receipt of that run). With several workers, the peak
can approach the sum of the largest concurrent kernels. Any worker count
above one needs its own memory admission before an actual-data run. The
provisional design's reservation budget is not implemented.

## Threads, not processes

Workers are threads. They share the admitted populations without a transport
layer, and no transport file can enter the object tree. The cost is the
interpreter lock. The workspace interpreter this was built and tested on is
CPython 3.14.4 with the lock enabled (`sysconfig` `Py_GIL_DISABLED` is `0`),
so only work that releases the lock, such as much of numpy's and pandas'
compiled code and file I/O, runs in parallel. Python-level loops inside
kernels serialize. A process pool would need an explicit transport for
contexts, results and registered kernels. That remains a separate design.

## Operational record

`run_graph(..., _concurrency_record={})` is filled with `max_workers`, the
counts `dispatched`, `declined`, `reused_context`, `verified_context`,
`discarded`, `submitted_at_turn` and `peak_submitted`, and per-node kernel
timings (`path` `inline`, `speculative` or `turn`, thread name, start and
finish). It is private, operational only, and enters no key, receipt, cache
record or manifest.

## Departures from the provisional design

| Provisional design (2026-09-19) | Implementation |
| --- | --- |
| Threads or processes undecided until measured | Threads. The lock limits speedup to lock-releasing work; processes need a transport. |
| A failed parallel run may publish a shorter prefix than the sequential one | The same prefix and the same exception: a result is read only at its node's turn. |
| An explicit memory reservation budget | A structural bound only (above). Kernel allocations are unbounded, as they are sequentially. |
| Attribution and transport measurements first, at 1/1000 and 1/15 | Not run. The identity proof is on fixtures. See [Evidence](#evidence). |

## Identity changes this branch makes

At every worker count, node keys, receipts, cache records, stored bytes and
the manifest key equal the sequential run's (tested below). The branch does
move one class of pin. The US native runtime binds whole modules into its
stage implementation hashes
(`microcosm.build.us_runtime.graph_implementation.implementation_manifest`),
including `microcosm.graph/executor.py`, the new
`microcosm.graph/scheduling.py`, and the inventory JSON itself. So all ten US
stage hashes move. A US native node whose kernel hashes one of these stages
gets a new key, and so does every node downstream of it, because
`_all_node_keys` derives each key from its inputs' keys. Those nodes will not
hit records a run at the base wrote. This is the repository's whole-module
identity rule working as intended, not a change in what any node computes.

Recomputed at the branch head by pointing `_package_roots` at a `git archive`
of `47960af43`'s package sources, in the same environment:

| Stage | Base `47960af43` | Head |
| --- | --- | --- |
| `acs_codec_2024` | `6c61215b59267f17317b1531aeff315f47318bedb072d725e8d5ec20c4b9d283` | `3291d2df9d57157ee6b6fa22aecb75b297f1599a3d28c51eb3adb2a6803e29bc` |
| `acs_housing_universe_2024` | `69fbe11a19ceaad3b4bc04d5be5bfd4f4cd12a6177f6b76535dc600d7724b33d` | `bfe9bcd8ec8a528619e41288d79d900a05b5e567e2275c3874ff71b94d8a4854` |
| `asec_codec_v4` | `ff7fcb3b95b73bef2ba3c9c022ebbc04c400ae1b61eb649501731ec354a05295` | `18593b3b6c409f51904ded37bff902541c027a239e15882e3f4d06bdfdb9941c` |
| `asec_prepared_v3` | `39a133f41c96186aada225cb1bc908ec71fefcd0152c122ff4b6cc792c6429b0` | `84a4f6ee1d5c91b107eb13ada47d32920b19c59518ae498f7963c58bcd7bce3b` |
| `assembly_harmonize` | `80c4a19b5e352135690c0748e679ab1947564be5eb439003fc3efbd93b7da7db` | `782ead6b1a724feac1826c08fae8411ec3ae1804586eebe6244121c5e6a72b16` |
| `assembly_prepare` | `add78163c07c8384663486171d44e6d6aacaa919679facbe623ce294e33fe45e` | `0d7391020c091d6382da5abd63fb0bd5970dda8dec6a68e421ebb90824a5bdc4` |
| `authenticated_survey_population_v1` | `32989bd3d285b2d80d5661f91a0aa28a36f558baaf7a372c700fbc1dd80fbe1e` | `ce324a033e1f6143e572ea918567b7dd718a3788b1f2b408682ec46b6d528f04` |
| `composed_asec_binding_v1` | `646dd7cd93e3d51b79a51a44ac1f1c41c03d697eb9022fd3b0d0bd112ce4bc6e` | `94abf0dae6d76075e79f20c487a813dba579e5aa17c6a3bfb04774b8fd1640d7` |
| `composed_population_v1` | `9fdd7b603348345bf0ff262cace687af73d58e2ca183b255bf413f419deb5527` | `3652891b79400996468c3be40cb3c0e79f4e45979e3951d98c4a7ec214c5210e` |
| `geography` | `cee1a147a2d272fc0e7c3ea66effb4d5ebb76509655e124b503be6f3f43e586a` | `ad6bb91d93727c14794733d33566786aafb204cee7c64667926791fab59ff2dd` |

A kernel built on `KernelBase` hashes only its own defining module
(`source_hash(type(self))` in `kernel.py`), so the keys of the toy and
acceptance kernels do not move.

## Evidence

Fixture-level, all run at the branch head in this worktree:

| Claim | Test |
| --- | --- |
| Manifest key, content-addressed projection, every node receipt, key, seed and hit flag, every attached population (values, dtypes, column order, weights, strata) and every relative path and byte in the object tree equal the sequential run's, at 2, 3, 4 and 8 workers, on eight graphs (small, chained, full, full with a failed gate, tolerance-scoped, structural, wide, typed artifacts) | `test_graph_parallel_executor.py::test_every_worker_count_reproduces_the_sequential_run_byte_for_byte` |
| Seeded per-node pauses produce at least two completion orders over six seeds at four workers, and every order folds to the sequential bytes | `test_seeded_completion_orders_all_fold_to_the_sequential_run` |
| Independent kernels really overlap: two kernels meet at a barrier neither can pass alone | `test_independent_siblings_actually_overlap`, `test_kernels_run_concurrently_not_merely_on_another_thread` |
| A warm rerun hits every record and dispatches nothing; a partial store resumes to the same bytes; `resume="forbid"` rewrites the same bytes | `test_warm_reruns_hit_every_record_and_dispatch_nothing`, `test_a_partial_store_resumes_to_the_same_bytes`, `test_forbidden_reuse_rewrites_the_same_bytes` |
| A failure raises the sequential exception with the sequential published prefix; a speculative sibling failure surfaces only at its own turn; a gate exception on a worker is still a failed verdict; context mutation on a worker is refused; an observer failure stops at the same node with the same store; a source changed by a worker evicts the run as it does sequentially | `test_a_failure_raises_the_sequential_refusal_and_publishes_its_prefix_only` and the five tests after it |
| The observer sees canonical order on the calling thread | `test_the_observer_sees_the_canonical_order_on_the_calling_thread` |
| A stale early projection is discarded and recomputed | `test_a_stale_early_projection_is_discarded_and_the_turn_recomputes` |
| Flag resolution, refusal before any write, environment opt-in, nothing in the portable manifest, scheduler bounds and readiness, join on close | the remaining tests in the file; an autouse fixture also fails any test that leaves a worker thread alive |
| Native nineteen-node financial graph (invented, source-issued fixture): at four workers, both manifests (the nine-node population prefix and the financial graph), every content receipt and every object SHA-256 equal the sequential run's, with native kernels on overlapping worker threads | `packages/microcosm-build/tests/test_us_graph_parallel_financial.py` |
| The existing graph suite, including the parity fixtures and acceptance replays, passes with `MICROCOSM_GRAPH_MAX_WORKERS=4` | `packages/microcosm-graph/tests` under the environment switch |

Timing on the same invented nineteen-node fixture (test-scale data, not the
1/1000 harness; wall and process CPU from `time.perf_counter` and
`getrusage`, each count cold into a fresh store, identity checked against the
first run):

| Run (in order, one process) | Workers | Wall s | Process CPU s | Kernel-call s (sum) | Kernel-busy wall s | Peak overlap | Identical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1 (first run, includes warm-up) | 1 | 49.04 | 38.11 | n/a | n/a | 1 | yes |
| A2 | 2 | 35.84 | 33.87 | n/a | n/a | 2 | yes |
| A3 | 4 | 35.27 | 33.73 | n/a | n/a | 2 | yes |
| A4 | 1 | 36.80 | 34.47 | n/a | n/a | 1 | yes |
| A5 | 2 | 34.60 | 33.23 | n/a | n/a | 2 | yes |
| A6 | 4 | 33.18 | 31.96 | n/a | n/a | 2 | yes |
| B1 | 1 | 32.73 | 31.71 | 10.09 | 10.09 | 1 | yes |
| B2 | 4 | 33.11 | 31.43 | 10.10 | 10.06 | 2 | yes |
| B3 | 1 | 32.86 | 31.61 | 10.51 | 10.51 | 1 | yes |
| B4 | 4 | 35.85 | 32.86 | 11.45 | 11.41 | 2 | yes |
| B5 | 1 | 34.64 | 32.54 | 10.81 | 10.81 | 1 | yes |
| B6 | 4 | 40.19 | 35.23 | 13.47 | 13.43 | 2 | yes |

Two processes (A and B), each building the invented sources once and then
running every count cold into its own store. "Identical" means that both
manifests' identities and every object SHA-256 equal those of the process's
first run. Kernel-call seconds sum each `invoke_kernel` call; kernel-busy
wall is the union of those intervals. The machine was shared with other
lanes' test runs, so differences of a few seconds are noise. Peak memory was
0.55 GB per process. There is no speedup on this fixture. After the first
run's warm-up, every count takes 33 to 40 s, and the two counts' ranges
overlap.

The graph explains why. A sequential run that recorded each kernel's time
(same fixture, one worker) spent 11.07 s in kernels. The longest path through
`compiled.predecessors`, weighted by those times, is 11.05 s, so almost no
kernel time can overlap at any worker count. The four heavy kernels form one
declared chain: `survey_predictors.asec_design_donor` (a FILTER, 2.28 s) is a
predecessor of `survey_predictors.source_projection` (2.40 s). That node is a
predecessor of `survey_predictors.asec_current_columns` (2.38 s), which feeds
the sequential `fit.000 -> fit.001 -> fit.002` chain. All of them precede
`survey_predictors.attach` (2.48 s). About two thirds of the wall time is
outside kernel calls, in the runner and in admission. Declared dependencies
do not change with the sampling fraction.

No 1/1000 actual-data measurement was run. When this lane checked, another
native survey run was active on the shared machine, and the lane measures
only when none is. What follows is a deduction, not a measurement. The
2026-09-23 native-line state note reads the 1/15 receipt as a node loop of
2,184 of 8,015 wall-seconds. Its four heaviest nodes were `attach` (497 s),
`asec_design_donor` (483 s), `source_projection` (473 s) and
`asec_current_columns` (466 s): 1,919 s, or 88% of the loop. They are the
four nodes of the chain above, so concurrency inside `run_graph` should save
at most a few percent of that run's wall time. It cannot shorten the 73%
outside the loop. The executor pays off only on graphs whose heavy kernels
sit on independent branches. Those may include the 51-node or 401-node
enrichment host graphs, but nothing has measured their shape.
