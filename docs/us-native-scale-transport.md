# US native build: receipt transport and per-node population retention

Design authority for the two hard limits the 2026-09-16 cost attribution found
in the native US graph build
(`experiments/us-native-executor-cost-attribution-20260916.md` on the
`microcosm-exec-profile` worktree, §5 and §9): a whole-roster receipt that
refuses above 6.1% of the US source, and a per-node population retention that
needs about 300 GiB at full source.

Every line number below is at `5ff889814`, the branch point, unless it names a
later head. Line numbers in the cost-attribution report are its own tree's and
are offset from these.

## 1. What refuses today, measured rather than inferred

The preparation receipt is **one JSON byte string**, built by
`prepare_authenticated_survey_population` and handed to `_encode` in a single
call (`survey_population_preparation.py:1679`) under
`MAX_PAYLOAD_BYTES = 64 * 1024**2` (`:50`), which refuses `PAYLOAD_LIMIT`
before each chunk append (`:104-109`). Twelve top-level keys; ten of them are
fixed-size. The two that are not are `"selection": _plan_document(plan)`
(`:1698`) and `"origins": origins` (`:1699`), which carry **one record per
source household and per person**:

| block | built at | per-row shape |
|---|---|---|
| `selection.selected`, `.excluded` | `_plan_document` `:240-253` | one `_value`-flattened plan row per supplied household |
| `origins.households` | `_origins` `:946` | one 7-field dict per stacked household |
| `origins.entities` | `_origins` `:870` | one `[new_id, channel, source_id]` row per row of **each** of the six US entities |
| `origins.persons.rows` | `_origins` `:965` | one 10-element row per stacked person |

A second, independent budget guards the same lists: `_bounded_append`
(`:816-819`) `_encode`s each row on its own and refuses `ORIGIN_LIMIT` against
the same 64 MiB constant. `_plan_document` opens its own budget (`:247`) and
`_origins` opens another (`:844`), so there are three 64 MiB ceilings on one
document, plus the final `_encode`.

**The size law, read off the pilot's own artifact rather than modelled.** The
1/1000 preparation receipt survives in `_recovered/pilot-runs/` — 1,136,063 B at
`native19-required-20260912/run/financial-artifacts/preparation.json`, byte
identical to `native19-v2`'s copy and to that run's
`graph-store/objects/04/0481…/payload.bin`, sha256 `34b362d85d2f06ac…`. It
records `supplied_households = 1587376` with 1,584 selected. Re-encoding it
through the repository's own canonical separators reproduces 1,136,063 exactly.
Decomposed: `origins` 718,800 B (households 269,000; entities 224,755; persons
210,924; storage_transitions 13,268; age normalization 762), `selection`
393,395 B (selected 385,122; excluded 7,250; cells 923), `producer` 21,135 B,
everything else 2,733 B. The blocks that grow with the draw are
1,096,821 B over 1,584 selected households:

> **bytes(f) = 39,242 + 692.4375 × 1,587,376 × f**

At f = 1 that is **1,099,197,911 B = 1.024 GiB, 16.4× the limit**. Encoding it
costs about 53 CPU-s at full source (the report's §5 measured 33.3 µs per source
household), so the ceiling is a transport-shape decision and not a performance
trade-off.

**Five ceilings, in the order they bind.** Each is computed from the same
committed artifact, or from the allocation payload's own measured 501,597 B at
1,584 households:

| # | ceiling | refusal | households | share of source |
|---|---|---|---|---|
| 1 | preparation `_encode` | `PAYLOAD_LIMIT` | 96,860 | **6.10%** |
| 2 | `_origins` row budget | `ORIGIN_LIMIT` | 150,916 | **9.51%** |
| 3 | allocation payload stream | `ALLOCATION_LIMIT` | 211,924 | 13.35% |
| 4 | `_plan_document` row budget | `ORIGIN_LIMIT` | 270,284 | 17.03% |
| 5 | allocation structural pre-check `ALLOCATION_MAX_BYTES // 128` | `ALLOCATION_LIMIT` | 524,288 | 33.03% |

A 1/10 build is 158,737 source households, so **#1 and #2 both bind and #3
does not**; a full build needs all five, because 524,288 is below the source's
own 1,587,376. #5 bounds the intermediate maps `allocation_instructions` builds
rather than the payload — its implied 127-byte row budget is below the payload's
220-byte structural overhead alone, so it can never be the binding *payload*
constraint — but it does refuse a full-source selection, so its ceiling moves
with the others and keeps its code.

**The store is not the constraint.** `ContentStore.put_bytes`
(`store.py:984`) and `put_frame` (`:919`) carry no byte cap, and `put_frame`
already writes one `.npy` per entity and column under
`objects/<key[:2]>/<key>` with a `meta.json` payload table recording sha256 and
size per file (`store.py:140-153`), re-verified on every read by
`_verified_meta` (`:227`). Every 64 MiB ceiling in this pipeline is owned by
the US runtime, not by the graph.

## 2. The new transport

### 2a. What must not move

`preparation_sha256` is `_sha(payload)` (`graph_survey_population.py:493`,
`:513`), it is folded through `_digest` into the CREATE and ALLOCATION node
params (`:1226`, `:1235`, `:1254`), and `params` reaches `node_key` through
`normative(node)` (`keys.py:288`). So the document's bytes reach every node key
and every store address. That is why the transport changes and the **canonical
byte stream does not**.

The stream is already defined independently of its materialisation.
`_chunks(value)` (`:97-102`) yields the canonical JSON of the document;
`_encode` (`:104-110`) concatenates those chunks under a bound, and `_digest`
(`:113-116`) hashes the same chunks under no bound. sha256 is a streaming hash
over exactly those bytes in exactly that order, so

> `_sha(_encode(document)) == _digest(document)`

for every document `_encode` accepts, and `_digest` keeps returning that value
for documents `_encode` refuses. The new transport digests the stream and
materialises it in bounded segments; it does not re-shape the document.

### 2b. The segmented, content-addressed receipt

`_segment_stream(document)` replaces the single `_encode` call for the two
whole-roster receipts. It walks `_chunks(document)` once and:

* accumulates into a segment buffer, and whenever appending the next chunk
  would exceed `MAX_SEGMENT_BYTES` (= the old `MAX_PAYLOAD_BYTES`, 64 MiB) it
  closes the segment — so **no accumulation in the encoder is ever larger than
  the value the old ceiling allowed**, and `PAYLOAD_LIMIT` still guards each
  one, with the same code, at the same number;
* records each closed segment's sha256 and size, and updates one running sha256
  over the whole stream;
* refuses `ROSTER_LIMIT` above `MAX_ROSTER_BYTES`, an explicit resource ceiling
  that is *not* a transport shape: it is the total the process is willing to
  spend, set far above full source (see §2d), and it is the only new refusal.

The result is a `_RosterReceipt` handle carrying `digest` (identical to
`_digest(document)`), `size`, and the segment table. The preparation path also
writes the segments to a content-addressed spill under the run's `snapshots`
directory — `<name>/<sha256>.segment` plus a small `<name>/header.json` naming
the protocol, the whole-stream digest, the size, and the ordered segment table —
which is the same header-plus-bodies shape `ContentStore.put_frame` already
uses, and which gives the receipt a re-verifiable on-disk form independent of
the graph store.

**The write surface, disclosed rather than left to be found.** This module
previously wrote nothing but `snapshots.mkdir`. The spill adds
`Path.write_bytes` for each segment and for `header.json`, and
`graph_implementation`'s `_RESOURCE_CALLS` (`:126-150`) covers *reads* —
`open`, `read_bytes`, `read_text`, `read_csv`, `load`, `files`, `joinpath`,
`import_module`, `exec`, `eval` and their neighbours — so a write is invisible
to `resource_accesses_sha256` by construction and no committed pin would notice
a redirected spill. Every component of the path is therefore checked here
instead: `root` is the snapshot root `_root()` already validated as symlink-free
and disjoint from the source root (`SNAPSHOT_LOCATION`), `name` must be a Python
identifier (`ROSTER_NAME`), each segment's own filename is the sha256 of its
bytes, the spill directory must not be a symlink (`ROSTER_SPILL_LOCATION`), and
an existing segment or header must be a regular file of the recorded size
(`ROSTER_SEGMENT_CHANGED`, `ROSTER_SPILL_LOCATION`). Presence is decided by
`lstat` and not by `Path.exists()`, because a broken symlink does not exist and
would be written straight through — a hole the first draft of this code had and
its own test caught.

`read_all()` reassembles, re-hashes each segment against its recorded digest
and the whole against `digest`, and refuses `ROSTER_LIMIT` above
`MAX_MATERIALIZED_BYTES`. The two consumers that need whole bytes — the
`preparation` and `allocation` opaque artifacts, which reach
`ContentStore.put_bytes` through `KernelResult.artifacts` — call it. So
`AuthenticatedSurveyPopulationPreparation.payload` stays `bytes` and **every
existing consumer is unchanged**: `_sha(entry[1])`, `type(value.payload) is
bytes`, the `CANDIDATE_MISMATCH` byte comparison, and the artifact publication
at `graph_survey_population.py:510`.

This is deliberately honest about what it does and does not buy. It removes
every ceiling that binds below full source and it keeps the encoder's peak
allocation at one segment. It does **not** remove the final materialisation:
1.02 GiB of bytes exists once, at full source, because
`KernelResult.artifacts` is a mapping of `bytes` and `put_bytes` takes whole
bytes. Removing that last copy needs a streaming artifact channel in
`microcosm-graph`, which is main-only and is §4's sibling. At full source
1.02 GiB against a ~26 GiB working set is not what stops a build.

### 2c. The row budgets

`_bounded_append`'s `ORIGIN_LIMIT` (ceilings #2 and #4) is a bound on the
Python row list, not on a transport. It keeps its code and its per-row
`_encode`, and its ceiling becomes `MAX_ROSTER_BYTES` — the same explicit
resource ceiling — so the refusal still fires, on the same rows, with the same
message, at a number a full-source build does not reach.

### 2d. The allocation payload

`_allocation_payload` (`graph_survey_population.py:344-369`) is already a
stream with a per-row bound: it emits `_bounded_json(_instruction_document(row),
4096)` per household into one bytearray with a pre-append check against
`ALLOCATION_MAX_BYTES` (`:362`) and a post-loop re-check (`:367`). Its bytes are
byte-identical to the canonical whole-tree encoding, which
`test_us_graph_survey_population.py:304` pins — an identity that holds only
because `"households"` sorts before all six metadata keys — so the same
streaming-digest argument applies.

It is built inside `SurveyPopulationAllocationKernel.run`, which has no
filesystem path and must not write to the store itself, so it takes the
in-memory half of §2b: bounded segments in a list, each guarded by
`ALLOCATION_LIMIT` at the unchanged 64 MiB, under `MAX_ROSTER_BYTES` in total,
materialised once. `allocation_sha256` (`:441`) is unchanged in derivation and
in value.

Ceiling #5, the `// 128` structural pre-check, moves to
`ALLOCATION_ROSTER_BYTES // 128` = 33,554,432 households. It bounds
`allocation_instructions`' intermediate maps, and its old 524,288 is **below**
the source's 1,587,376, so a full-source selection refused there too. The code
is unchanged.

**One refusal narrows, and this design says so rather than leaving it to be
found.** `ALLOCATION_LIMIT` fired when the *whole* payload passed 64 MiB. Under
a segmented stream that condition no longer arises: a segment closes instead. It
still fires, with the same code and the same expression, on the condition a
segmented stream can still reach — **one instruction row plus the reserved tail
larger than one whole segment**. `test_us_graph_survey_population.py` drives both
that refusal and the new total ceiling, and asserts the payload's bytes are
unchanged at four different segment sizes down to the smallest one that holds a
row. The same narrowing applies to the preparation receipt's `PAYLOAD_LIMIT`:
`_chunks` yields one JSON token at a time and a segment never splits one, so the
refusal now means "one token larger than one segment", and because
`_check_scalars` has already bounded every string to 1 MiB the shipped 64 MiB
segment always holds any token. Both are tested at a segment size below the
longest token.

### 2e. Size law after the change

`MAX_ROSTER_BYTES` is set to `64 * MAX_SEGMENT_BYTES` = 4 GiB, which is
3.9× a full-source preparation receipt and 7.6× a full-source allocation
payload (measured 316.665 B per household × 1,587,376 = 502.6 MiB). The
refusals therefore still exist, still fail closed, and no longer bind below
full source:

| | before | after |
|---|---|---|
| preparation receipt | 96,860 hh (6.10%) | 6,206,000 hh (391% of source) |
| `_origins` rows | 150,916 hh (9.51%) | 9,658,000 hh |
| allocation payload | 211,924 hh (13.35%) | 13,563,000 hh |
| `_plan_document` rows | 270,284 hh (17.03%) | 17,298,000 hh |
| allocation map pre-check | 524,288 hh (33.03%) | 33,554,432 hh |

A fourth whole-roster receipt the cost attribution did not name is in the same
module: `_current_survey_wage_projection` builds one row per selected wage
earner and encoded it the same way. It takes the same transport. It does not
bind at 1/10 and does bind at full source.

## 3. Retention and the frame seal

### 3a. What the observer costs and where

`run_graph` calls `_population_observer(node_id, _observer_snapshot(updated))`
at `executor.py:2773-2774`, under the single guard
`if _population_observer is not None:`. That is **outside** the cache-load
attempt (`if resume != "forbid":`, `:2575`), outside the cold path
(`if result is None:`, `:2607`) and outside the store write (`if not hit:`,
`:2776`), so every reached node pays it, cache hits included; only the
unreached-node `continue` (`:2510-2524`) skips it. `_observer_snapshot`
(`:356-408`) detaches by round-tripping every entity and link table plus the
strata through `pickle.loads(pickle.dumps(..., protocol=5))` (`:373`) — a full
independent copy, measured by the cost-attribution report at **9.6–9.8 bytes
per cell per snapshot**.

The financial runner retains all of them: `observed[node_id] = population`
(`graph_atomic_survey_financial.py:1644`), never cleared, read at `:1656`
(node ids), `:1796`, `:1818`, `:1821`, `:1824`, `:1944`, `:1948`, `:1974`
(the objects), `:1897` (object *identity*) and `:1945` (the stamp alone). It
reaches the frozen `_FinancialRunState.node_populations` field (`:127`) through
`_issue_run` and `_node_population_seals` (`:500-515`), and `_pure_run`
(`:676-691`) re-stamps every retained population on **every** `checked_view()`.

### 3b. Why the snapshot cannot be addressed to the store

The obvious move — write each snapshot to the content store and keep the key —
is refused by the store's own normalisation, in three independent ways, and this
design records them rather than discovering them later:

* `store.py:456` zeroes the values beneath a null mask (`values[mask] = 0`),
  while `survey_atomic_geography._population_stamp` (`:230`) deliberately folds
  the physical storage parts *under* the mask — its docstring says so at
  `:231-237`. A store round trip therefore changes every stamp.
* `store.py:619` refuses a `MultiIndex` outright, and
  `test_graph_executor.py:4254-4255` builds one into the observed frame.
* `store.py:486-487` refuses `CategoricalDtype` and `DatetimeTZDtype`, and
  `test_graph_executor.py:4262` observes a categorical column.

There is also no store API that accepts a `Population` (only `put_frame` for a
`Frame` and `put_bytes` for bytes), and `DataFrame.attrs` — asserted preserved
at `test_graph_executor.py:4320-4321` — is not persisted by `_write_frame`.
**A store-backed snapshot is not a drop-in for the pickle.** Anything that
spills a snapshot must spill the same pickle bytes the executor already trusts
for detachment.

### 3c. Object identity is pinned, so re-materialisation is not free

`result.financial_population is observed[final_node]`
(`graph_atomic_survey_financial.py:1897`) is an identity check, not an
equality check, and the completion host has two more of its own. A policy that
dropped a population and rebuilt it from a spill would produce a different
object and fail these. So the retention policy is **declared-consumer**: the
nodes whose objects a caller holds are named, retained in memory, and the rest
may be spilled. For the base financial run those are the final node,
`financial.ATTACH_NODE`, and — when the tax rebase is enabled — the property
population.

### 3d. What still needs every population, and the question it raises

Two loops need the whole `Population` object, not a seal:
`atomic.same_replayed_population(population, observed[node_id])` at `:1796` and
`:1948`. They compare the executor's observation against the runner's own
independent replay, field by field and byte by byte
(`survey_population_replay.py:132`, `:179`). The replay cannot be built until
`run_graph` has returned, because it reads the manifest's loaded artifacts, so
between the observation and the comparison **nineteen detached populations must
exist somewhere** — RAM (~290 GiB at full source) or disk (~285 GiB). There is
no third place, and choosing to replace the object comparison with a content
seal is a change to a verification contract, not an optimisation. §5 puts that
to the owner as a question.

### 3e. The seal: a vectorised per-column pass, proven equal

`survey_population_preparation._frame_identity` is the only remaining per-cell
frame walker in the US runtime. Its cost was literally

```python
for value in series:
    digest.update(_frame_cell_encode(value))
    digest.update(b"\n")
```

— one `_frame_cell_encode` call and two `digest.update` calls per cell, over a
quantity that is a pure function of the cell values plus dtype metadata. (The
two neighbours that look similar are already vectorised and are left alone:
`asec_current_money_source._series_digest` (`:98`) dumps `_data`/`_mask` bytes
for numeric dtypes and caches string tokens, and
`asec_2024_native_population._frame_identity` (`:184`) delegates to it.)

The replacement builds, per column, **the exact byte string that loop would
have fed the digest**, and updates the digest once:

| dtype | how | why it is exactly equal |
|---|---|---|
| `float64` | `_float_cells`: a fixed-offset `(n, 37)` uint8 matrix with a per-position keep mask, cut down with one boolean gather | the record is `["float","` + optional `-` + `0x1.`/`0x0.` + 13 mantissa nibbles + `p` + sign + up to 4 exponent digits + `"]`; the mask deletes the parts a value does not spell, so `float.hex()`'s variable spellings come out of one padded matrix |
| `int8`…`uint64` | `_integer_cells`: `"\n".join(map(str, values.tolist()))` | `str` on a Python int **is** `int.__repr__`, which is the fast path's encoder; every integer-dtype value lies inside `-2**63 <= v < 2**64` and spells ≤ 20 bytes, so neither `_encode` nor `PAYLOAD_LIMIT` is reachable |
| `Int8`…`UInt64` | `_nullable_integer_cells`: the same, with `pd.NA` positions overwritten by `null` | `_cell` maps `pd.NA` to `None` and `_frame_cell_encode` spells `None` as `null`; unsigned widths keep an unsigned carrier so `UInt64` above `2**63 - 1` still spells |
| `bool`, `boolean` | `_boolean_cells`: index into the three spellings | three possible cells |
| `string` | `_string_cells`: `pd.factorize`, then `_frame_cell_encode` once per distinct value | `StringDtype` compares and hashes exactly, so grouping equal values cannot merge two whose encodings differ |
| anything else | the cell-at-a-time walk, unchanged | `object` and `category` deliberately do **not** qualify: `1`, `True` and `1.0` are equal and hash alike there while `_frame_cell_encode` spells them three different ways |

`_cell` refuses a non-finite cell with `FRAME_NONFINITE`; the float branch
raises the same code for the column rather than at the first offending cell,
and the digest is discarded either way.

**The equality is proved against an independent oracle that already existed.**
`packages/microcosm-build/tests/test_us_survey_frame_identity_encoding.py`
carries `_legacy_frame_preimage`, a verbatim predecessor traversal, and
`_assert_exact_frame_preimage`, which records every `digest.update` call through
a patched `hashlib.sha256` and asserts the **concatenated byte preimage** — not
merely the digest — equals the oracle's. That is a byte-for-byte proof, and it
is joined by an exhaustive float proof and a per-dtype sweep this lane adds.

## 4. What `run_graph(population_retention=...)` on main would need

`#901` recorded that the shared executor has no retention mode, and that is
still true at this head: there is no eviction, pruning or population dropping in
`executor.py` or `store.py`, and the only occurrences of the name
`population_retention` in the tree are historical notes in `experiments/` and
`out.md` describing a dropped branch. The executor's own `populations` dict is
keyed by **version**, not node, and a `StructuralDelta.NONE` node overwrites its
entry (`:2769`), so the executor itself already retains one population per
version; the per-node retention is entirely the observer's.

A `population_retention` keyword would have to carry these, all of them
established above:

1. **A mode that skips the detachment, not just the retention.** The cost is
   `pickle.dumps` + `pickle.loads` per node (`:373`). An observer that only
   seals needs no detached copy at all, because sealing is read-only. The
   executor would hand it the live population and the mode would be the
   observer's declaration that it will not retain or mutate it.
   `test_graph_executor.py:4368-4374` already pins the converse — with no
   observer, no snapshot may be allocated — so the shape of that test is the
   shape of the new one.
2. **A declared-consumer roster.** `_observer_snapshot`'s detachment exists so
   the observer may mutate what it keeps (`test_graph_executor.py:4183-4248`
   asserts a mutated snapshot moves no node key, no manifest and nothing in the
   store). A retention mode that hands out a read-only view instead would make
   that mutation raise — *more* closed, but a behaviour change to a documented
   contract in `microcosm-graph`, so it belongs to a main-only change with its
   own test rather than to a country runner.
3. **Invocation invariants that do not move.** Exactly once per reached node, in
   `compiled.order`, on cache hits and cold execution alike
   (`test_graph_executor.py:4153`, `:4166-4167`); an observer's own exception
   still refuses the run with the observer's exception type (`:4172-4179`);
   `ATOMIC_OBSERVER_ROSTER` and `COMPLETION_OBSERVER_ROSTER` still refuse any
   gap or reorder.
4. **A precedent in the same package.** `graph_survey_puf55` already declines to
   retain everything: its issued run entry "retains the final output, not all 245
   executor-observed snapshots" (`graph_survey_puf55.py:638-639`). The financial
   runner's `node_populations=observed` is the outlier among the six observers in
   this runtime, and the shape it would move to already exists a few files away.
5. **A statement about `_pure_run`.** `state.node_populations` is re-stamped on
   every `checked_view()`, so a retention mode that drops populations must say
   what `FINANCIAL_NODE_POPULATION_CHANGED` means afterwards. Today
   `_node_population_seals` already returns `()` when there are no observations
   and no completion boundary (`:502-503`), which is the existing shape of
   "nothing retained, nothing to re-stamp".

Item 1 is the whole memory wall and most of the avoidable CPU. It is main-only,
it lands in `microcosm-graph`, and the executor hunks of PR #938 are its
neighbours.

## 5. The ceilings this lane did not touch, and where they sit

A transport ceiling is not the only kind. The 19-node path carries a second
family — row-count and row-byte bounds — and this section records where each one
sits so the next lane does not rediscover them one build at a time. The counts
come from the recovered 1/1000 artifact's own rosters: 1,584 selected households
carry 1,584 stacked households and 3,464 stacked persons, i.e. 2.187 persons per
household, and the combined clone doubles both.

| bound | value | binds at |
|---|---|---|
| `survey_population_domains.MAX_HOUSEHOLDS` | 100,000 | **never** — it bounds one `classify_households` batch, and `survey_catalogue_selection` streams batches of at most `_BATCH_HOUSEHOLDS = 10,000` households or `_BATCH_PEOPLE = 100,000` people (`:187`, `:197-201`) |
| `survey_population_domains.MAX_TOTAL_MEMBERS` | 1,000,000 | **never**, for the same reason |
| `survey_population_domains.MAX_MEMBERS` | 20 | per household, unrelated to the fraction |
| `acs_person_coverage_columns.MAX_SELECTED_ROWS` | 1,000,000 | above 1/10; 347,137 stacked persons at 1/10, 3,471,383 at full source |
| `asec_current_money.MAX_PERSONS` | 1,000,000 | above 1/10, same counts |
| `acs_pums.MAX_EXACT_HOUSEHOLDS` / `MAX_EXACT_PERSON_ROWS` | 1,000,000 | above 1/10; 1,587,376 households at full source |
| `survey_observed_age.MAX_ROWS` | 2,000,000 | above 1/10; ~694,000 cloned persons at 1/10, ~6,943,000 at full source |
| `survey_origin_budget.MAX_GROUPS` | 1,000,000 | not established — the group count was not derived here |
| `graph_survey_age_artifact.MAX_PEOPLE` | 10,000,000 | above full source's ~6.94M cloned persons, so not at all |

**So a 1/10 build meets no ceiling this lane did not lift, and a full-source
build meets at least four more** — all of them row counts rather than transport
shapes, and all of them in the 1,000,000–2,000,000 range that a 1.59M-household
source crosses by construction. They are a separate change with a separate
argument, and this lane does not touch them.

## 6. Open, for the owner

1. **Columnar format for the receipt bodies.** This design uses the store's own
   existing shape — a `header.json` payload table plus content-addressed bodies,
   as `put_frame` already writes `.npy` files — and adds no dependency. A true
   Arrow/parquet body would need `pyarrow` in the workspace and would change the
   canonical byte stream, and therefore `preparation_sha256` and every node key.
2. **Whether the retention policy is a run option or the only behaviour.** §3d
   shows that nineteen detached populations must exist between the observation
   and the replay comparison unless the object comparison is replaced by a
   content seal. That is a verification-contract decision, and §4's item 4 shows
   the runtime has already made it once, the other way, in `graph_survey_puf55`.
3. **Whether the roster spill should be readable by anything but this module.**
   It is written, verified, and then only reassembled in the same call today. A
   consumer that wanted to stream it — a future `put_bytes` that takes an
   iterator, or an auditor re-deriving `preparation_sha256` without loading
   1 GiB — would read `header.json` and the segments it names, which is why the
   header exists at all.
