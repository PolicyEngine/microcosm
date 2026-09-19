# Concurrent graph executor: design and evidence requirements

Status: provisional design, 2026-09-19. No concurrent executor has been
implemented or measured by this note. The runtime code audited here is the
code at `16c8e78d2`, also unchanged in the lane's initial journal commit.
References below are to that audited version; their line numbers must be
refreshed if the implementation moves. The baseline attribution, worker
boundary, memory reservations, and population sizes remain measurements to
complete, not established results.

## Existing guarantees

| Guarantee | Mechanism at the audited head |
| --- | --- |
| Canonical execution order | `compile_graph` computes depth and sorts by `(depth, node id)` in [decl.py](../packages/microcosm-graph/src/microcosm/graph/decl.py#L816-L833). The executor iterates that order in [executor.py:2676](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2676). |
| Every node key precedes every execution | `_all_node_keys` walks the whole order, obtains and validates each kernel implementation hash, and derives keys from previously derived input keys in [executor.py:2205](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2205-L2229). `_execute_graph` completes it, all typed contracts, and required replay preflight before entering the loop at [executor.py:2665](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2665-L2676). |
| Complete declared dependencies | An ordinary node depends on its population version and its input-column owners. A structural successor depends on its base and **all** ordinary members of that base; typed artifact producers join the same predecessor sets in [decl.py:732](../packages/microcosm-graph/src/microcosm/graph/decl.py#L732-L814). Row masks must themselves be declared input columns at [decl.py:492](../packages/microcosm-graph/src/microcosm/graph/decl.py#L492-L503). |
| Unavailable inputs propagate without invented products | `_blocked_by` names an unreached causal predecessor or a producer of an unavailable typed artifact, sorts blockers, and deliberately does not block on an ordinary failed gate verdict in [executor.py:2232](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2232-L2258). `_unreached_node` creates only an exceptional receipt/cache record, authenticates matching blocker provenance on a hit, and creates no population in [executor.py:2261](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2261-L2357). The loop records it and continues at [executor.py:2688](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2688-L2703). |
| A failed gate can remain graph evidence | The kernel exception arm converts a gate exception into a failed gate result, while ordinary kernel exceptions refuse the run in [executor.py:2804](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2804-L2814). A gate's unavailable byte outputs are separately recorded at [executor.py:150](../packages/microcosm-graph/src/microcosm/graph/executor.py#L150-L163). |
| Required replay fails before execution | `_preflight_require` validates every record, predecessor/blocker provenance, tolerance writer contract, and its referenced products, then raises for the complete collected miss list in [executor.py:2360](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2360-L2415). It runs before populations or node execution exist. Per-node `require` checks still defend subsequent races at [executor.py:2783](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2783-L2784). |
| Admitted populations precede callbacks and persistence | The result is validated and patched at [executor.py:2851](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2851-L2896), then attached to its version, observed, persisted if cold, and finally represented by a complete node receipt at [executor.py:2955](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2955-L3016). |
| Node cache records follow all their products | `_write_node` writes sorted columns, a structural frame when applicable, replacement weights when present, sorted opaque artifacts, then the node record in [executor.py:1593](../packages/microcosm-graph/src/microcosm/graph/executor.py#L1593-L1703). |
| Atomic objects have a complete publication ledger | `ContentStore._put` builds in a unique temporary directory, hashes payloads, fsyncs, atomically publishes, and notes only actual publications in [store.py:791](../packages/microcosm-graph/src/microcosm/graph/store.py#L791-L847). Existing verified objects are excluded; write-only replacements are included. `recording_writes` is instance-scoped, including concurrent publications through that instance, at [store.py:720](../packages/microcosm-graph/src/microcosm/graph/store.py#L720-L751). |
| Failed runs remain resumable only against unchanged sources | `run_graph` owns the ledger for the whole call and catches `BaseException` before settling at [executor.py:2597](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2597-L2615). `_settle_failed_run` re-derives every source without the identity cache; unchanged sources preserve completed work, changed or unverified sources evict the entire written set, retaining the original exception with a note at [executor.py:2468](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2468-L2529). |
| Successful return also re-authenticates sources | Source keys are re-derived without the per-run cache, the source record is settled, changed-source writes are evicted, and only then is the manifest constructed in [executor.py:3050](../packages/microcosm-graph/src/microcosm/graph/executor.py#L3050-L3094). |

The current source-identity cache already reuses a source content key within
one `run_graph` call when its complete stat signature matches the signatures
on both sides of the original read. It includes directory membership and
symlink target identity. It does not make a persistent authentication receipt,
nor eliminate the mandatory cache-bypassing final read:
[executor.py:2124](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2124-L2202).
#938 (`8c44daa52`) merged graph-only source/digest optimizations and the
manifest verification-epoch carrier. Its changed-file inventory contains no
US runtime module. The carrier accepts caller-owned counts; it does not
perform US source verification. The separate native branch's US epoch
wrappers (#935) are absent from this main-based head. Attribution must read
the actual measured runtime's implementation and distinguish its repeated
work from the graph optimizations already present here.

## Inputs and the store

`populations: dict[str, Population]` is the executor's in-memory source of
population inputs. CREATE has no incumbent; an ordinary node reads the
current population for `compiled.versions[node_id]`; another structural
node reads its base at
[executor.py:2673](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2673-L2712).
The population for an ordinary version is cumulative: each admitted ordinary
node replaces that version's value with its newly patched population.

Kernels receive a projected `KernelContext`, not that complete Population.
`_project_context` selects declared columns plus structural identifiers,
rewrite incumbents and materialized EXPAND claims, freezes copied tables,
supplies appropriate weights/strata, binds declared source paths, and creates
an RNG from the node key at
[executor.py:637](../packages/microcosm-graph/src/microcosm/graph/executor.py#L637-L751).
Input context digests before and after execution refuse mutation at
[executor.py:2803](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2803-L2831).

Typed artifact edges authenticate the producer receipt and descriptor before
cache lookup. Their actual bytes are read from the store only for a cold
kernel, at
[executor.py:2727](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2727-L2747)
and [executor.py:2786](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2786-L2802).
A cache hit reconstructs and validates its result against the incumbent at
[executor.py:2754](../packages/microcosm-graph/src/microcosm/graph/executor.py#L2754-L2784).
Ordinary populations therefore cannot be reconstructed by loading only their
version's original stored frame: the intervening ordinary-column patches
also matter. Structural successor node keys already bind those base members
in [keys.py:213](../packages/microcosm-graph/src/microcosm/graph/keys.py#L213-L225).

## Proposed execution and admission boundary

The public runtime option will be `run_graph(..., max_workers=1)`. The default
must retain the existing path and operational semantics. No declaration,
kernel interface, normative key projection, or producer pin change is
proposed. Amendment 26 should be recorded only with the implemented and
tested runtime contract.

Readiness means that **every dependency's complete receipt exists**. A future
that finished its kernel is not a dependency receipt. The coordinator first
checks blockers using only admitted predecessor receipts. Unreached nodes
continue to acquire their exact exceptional receipts without running a
kernel. Required replay preflight still completes before any worker is
started or any node is admitted.

Concurrent execution must be separated from deterministic admission. Ready
workers may produce results in different orders, but one coordinator must
apply results, update cumulative populations, invoke observers, write store
objects, and admit receipts in `compiled.order`. A completed later result can
wait in a bounded result queue. Contexts contain only declared inputs, whose
writers have already been admitted; independent ordinary siblings cannot
change those declared inputs. The coordinator applies each ordinary result
to the latest cumulative incumbent, not the older population used when that
worker started.

This ordering is necessary for byte identity, not merely convenient locking:

- Two independent ordinary nodes can share a version. Running the entire
  existing loop concurrently lets each replace the version from an older
  incumbent and lose the other's columns.
- `_patch_columns` inserts newly owned columns into the frame in admission
  order at
  [population.py:1764](../packages/microcosm-graph/src/microcosm/graph/population.py#L1764-L1848).
  A later structural frame serializes the resulting column order.
- `patch` permits `frame_mass_log_append` even for ordinary nodes, and the
  append preserves execution order at
  [population.py:1151](../packages/microcosm-graph/src/microcosm/graph/population.py#L1151)
  and [population.py:2117](../packages/microcosm-graph/src/microcosm/graph/population.py#L2117-L2221).
- The observer currently sees the complete cumulative population at each
  canonical node boundary. Merely protecting callbacks with a mutex while
  invoking them in completion order would change those observations.

Consequently the agreement sweep must vary legal worker dispatch/completion
orders while retaining the canonical result fold. There is no claim that
arbitrary topological **application** orders are byte-commutative at this
head. Structural successors already wait for all ordinary members of their
base, so their canonical frames include all those members.

### Worker transport: pending measurement

Threads can share the admitted in-memory populations and require no process
transport. Python-heavy kernels may receive little CPU parallel speedup from
them; native operations that release the interpreter lock need measurement.
Sharing a kernel instance also needs review because the registry currently
returns the same registered object on each call.

Processes can run Python on multiple cores, but a complete Population or
KernelContext includes records and mapping proxies that need an explicit
transport decision. Options are a trusted, ephemeral serialization of the
projected context/result, or reconstruction from authenticated store products.
Loading only a structural frame misses ordinary patches, and writing worker
transport objects into the content-addressed object tree would violate the
required identical store inventory. Any transport files must remain outside
that tree and be cleaned up.

Select the boundary only after recording, at 1/1000, resident Population
bytes, projected-context bytes, serialized sizes, serialization and restore
CPU/wall costs, and observed kernel CPU concurrency. The 19-node runtime and
harness must be available against the audited code before that decision is
called measured. No population weight or selected backend is asserted here.

### Failure, workers, and the publication ledger

The proposed coordinator-only writer keeps workers from publishing store
objects. On the first observed hard failure, scheduling stops; work that has
not started is canceled. Already running workers finish and are joined,
without admitting their results after failure. The first failure is retained
while joining, including when another worker subsequently fails. Gate
exceptions that become ordinary graph evidence retain that treatment.

Joining occurs **inside** the outer store-ledger lifetime and before
`_settle_failed_run`. There must be no worker that can mutate a source after
the failure source check, and no publication that can arrive after its
written set was copied. The proof for the proposed boundary is simple:
workers never publish; only coordinator admissions add ledger entries; after
failure admissions stop; worker joining changes none of those entries;
settlement sees exactly all publications that occurred in this run. Stable
sources preserve the successful prefix for resume; changed or unverified
sources evict all of that run's publications. Required replay after a
partial failure still fails preflight if any node is absent.

The same publication set need not have the same size as a failed sequential
run: observing a later sibling's failure early can stop admission before the
sequential path would have reached it. The invariant is that settlement
sees every actual publication, with no late writes and no inherited
pre-existing hits misclassified as this run's work. This distinction must
be explicit in the failure tests and amendment.

No consumer relies on directory insertion order. Consumers address objects
by derived keys, and readiness requires the producer receipt, which is only
admitted after its cache record and products exist. Keeping coordinator
writes in canonical order additionally preserves today's cross-node write
ordering. Within each node the existing products-before-record order stays
unchanged.

### Observers

[Amendment 24](graph-acceptance.md#L494-L513) uses a detached snapshot on both
cold execution and restored cache hits before persistence. It detaches table
and object-cell storage, axis/category buffers, schema and link records,
weights and design anchors, metadata, owners, and mass ledgers;
`_observer_snapshot` implements this at
[executor.py:355](../packages/microcosm-graph/src/microcosm/graph/executor.py#L355-L407).
Snapshots may be retained or mutated without changing computation or store
bytes. Unreached nodes have nothing to observe; absent observers allocate no
snapshot. An observer exception refuses the run.

[Amendment 25](graph-acceptance.md#L515-L527) lets a caller assert its observer
will neither retain nor mutate a live Population using
`_population_observer_detach=False`. This intentionally withdraws execution
and persistence isolation; it does not make mutation safe. Both modes must
run only on the coordinator, in canonical admission order, before the same
node's persistence. No two callbacks overlap. Neither worker count nor
observer mode enters keys or stored receipts.

### Memory admission: pending measured coefficients

`max_workers` is an upper bound, not permission to launch that many
unbudgeted populations. The concurrent path needs an explicit runtime memory
budget, with reservations covering live admitted populations, projected
contexts, input artifact bytes, worker/result buffers, structural frame
copies, and the observer's optional snapshot and temporary serialization
buffer. Pending completed results count against the same budget until
admitted or discarded. Retained snapshots belong to the caller and cannot
be bounded by a worker count alone.

For a proposed new worker, scheduling requires both a free worker slot and
`resident reservation + in-flight reservations + proposed reservation <=
budget`. A task larger than the available budget needs a clear refusal,
rather than a scheduler that waits forever. At least one running worker can
finish without needing another worker's slot or an unreserved result
allocation. Reservations must survive until transport, result application,
and observation have released their temporaries. Expansion and model
training can allocate more than input size; coefficients or per-node
estimates must come from the measured harness, with conservative headroom.
A bytes-in counter alone does not guarantee a hard RSS limit for arbitrary
kernel code, so the implementation must state the limit of its guarantee.

The shared-machine measurement gate is separate: a measurement starts only
when `vm_stat` reports more than 45 GB available, records its exact PID, and
runs under `ulimit -t 21600`. The 1/15 comparison is conditional on that gate.

## Identity claim and evidence still required

The exact claim is equality of the manifest **key**, every content-addressed
receipt payload, and every relative file and file byte in the store object
tree at 1, 2, and 4 workers. Full operational manifest JSON is deliberately
not byte-identical across ordinary reruns today: `NodeReceipt.hit` and
`wall_time` are run fields, release decision-derived `outcome` is excluded
from content identity, and manifest host/timestamps are operational fields.
See [manifest.py:403](../packages/microcosm-graph/src/microcosm/graph/manifest.py#L403-L428),
[manifest.py:432](../packages/microcosm-graph/src/microcosm/graph/manifest.py#L432-L462),
and [manifest.py:709](../packages/microcosm-graph/src/microcosm/graph/manifest.py#L709-L734).
`content_addressed` sorts node ids and the manifest key hashes that projection
at [manifest.py:579](../packages/microcosm-graph/src/microcosm/graph/manifest.py#L579-L590)
and [manifest.py:660](../packages/microcosm-graph/src/microcosm/graph/manifest.py#L660-L661).
Stored node records contain the computation receipt and omit operational
timing/cache-hit fields.

The intended proof is induction over canonical admissions: keys/contracts
are unchanged; admitted predecessors and declared source bytes are unchanged;
a conforming deterministic kernel receives the same projected input and
key-derived RNG seed; validation and canonical application yield the same
population and computation receipt; unchanged store serialization publishes
the same bytes; the next admission therefore has the same premise. The
detached observer cannot affect this induction. The live observer option
retains amendment 25's caller obligation. Platform- or tolerance-scoped
kernels retain their existing declared numeric contracts; a worker backend
must not silently change their math libraries' execution behavior.

Required evidence before treating this note as final:

1. Cold plus required-replay attribution at 1/1000, using the transport lane's
   0.25-second sampler, with each outside-graph bucket tied to read code and
   classified by its actual reusable input/output identity.
2. A measured process/thread/transport decision and population sizes; memoized
   buckets with second-access store-hit and identical-byte tests, preserving
   each refusal condition.
3. Existing graph tests unchanged at one worker; acceptance graphs at 1/2/4
   workers comparing manifest keys, content receipt bytes, and complete store
   object inventories; seeded dispatch/completion agreement sweeps.
4. Failure tests with siblings in flight, observer failure, source changes,
   exact publication ledger settlement, and required replay after partial
   failure. A completed but unadmitted worker must publish nothing.
5. Wall, CPU, peak RSS, node-loop/outside split and replay identity for the
   19-node 1/1000 runs, followed by 1/15 at 1/2 workers if memory allows.
   Refusals and recovered exception causes are evidence, not suppressed runs.
6. Amendment 26, unchanged `decl.py` and `kernel.py` interface lock, moved-pin
   table (empty only if no pinned input changes), CI-equivalent validation,
   changelog, committed progress, and a draft PR.
