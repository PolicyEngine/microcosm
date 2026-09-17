# Build lane: the native build's transport ceiling and per-node retention

Branch `native-scale-transport`, worktree
`~/PolicyEngine/_worktrees/microcosm-native-scale`, stacked on
`native-verify-once` (#935). Draft PR
**[PolicyEngine/microcosm#945](https://github.com/PolicyEngine/microcosm/pull/945)**
— draft, and it stays draft.

Every line number is at this branch's base `5ff889814` unless it names another
head. Line numbers in the 2026-09-16 cost attribution are its own tree's and are
offset from these; each one this report relies on was re-derived here.

## 1. Baseline (requirement 1) — and it settles the question the attribution left open

One uncapped-by-node run of the 19-node financial graph at 1/1000, at the branch
point, in the throwaway worktree
`~/PolicyEngine/_worktrees/microcosm-native-scale-baseline`, under the same stack
sampler the verify-once lane used plus a current-resident-size trace sampled
every second through `libproc`'s `proc_pidinfo` (`ru_maxrss` is a high-water mark
and a retention slope needs the trace). pid 70342 / pgid 70342, session leader;
`RLIMIT_CPU` hard 9,060 s with the harness's own soft ceiling at 9,000 CPU-s,
12,000 wall-s, 48 GiB; launched only after `vm_stat` showed 69.9 GB available.
Status `COMPLETED_NINETEEN_NODE`.

| | |
|---|---|
| wall | 2,048.93 s |
| **CPU** | **2,028.69 s** |
| peak RSS | 13,157,728,256 B = 13.16 GB / 12.25 GiB |
| loadavg | start [6.68, 6.97, 6.40] → end [10.75, 9.68, 8.66] |

That is within 0.9% of the verify-once lane's own 19-node figure (2,010.07 CPU-s,
2,016.08 s wall, 13.31 GB), which is the expected agreement for two runs of the
same code on a shared machine, and it confirms 2,010–2,029 CPU-s as the number to
beat rather than v4's 5,278.6.

**The runner time outside the node loop.** The harness timed the
`run_atomic_survey_financial` call on its own and summed the per-node
`wall_time_s` the executor writes at `executor.py:2821`, which spans the whole
node-loop body:

| | wall seconds | share of the runner call |
|---|---|---|
| the runner call | 2,045.91 | 100% |
| financial node loop, 19 nodes | 277.36 | 13.6% |
| prefix node loop, 9 nodes | 55.93 | 2.7% |
| **outside both node loops** | **1,712.62** | **83.7%** |
| process wall outside the runner call | 3.02 | — |

**So the per-node work is not what a build of this size costs — the runner's own
source authentication, reconstruction and verification is.** Inside the
financial loop, the four ASEC predictor nodes are 267.99 s of the 277.36 s and
the other fifteen are 9.37 s:

| node | wall s | | node | wall s |
|---|---|---|---|---|
| `survey_predictors.attach` | 72.183 | | `survey_population.create` | 0.668 |
| `survey_predictors.asec_design_donor` | 66.882 | | `survey_population.allocate` | 0.660 |
| `survey_predictors.asec_current_columns` | 64.514 | | `geography.derive` | 0.572 |
| `survey_predictors.source_projection` | 64.406 | | `geography.assign` | 0.561 |
| `combined_survey_puf_support_clone` | 2.799 | | `geography.gate` | 0.557 |
| `survey_predictors.apply.000` | 0.695 | | `survey_population.observed_geography` | 0.292 |

**The retention slope.** It is not visible at this fraction, and that is the
honest answer rather than a gap. Nineteen snapshots at the attribution's measured
9.6–9.8 B per cell over this frame's 1.70e6 cells is **0.31 GiB**, and the RSS
trace's own sawtooth is 1–2 GB: the peak is 13.16 GB at t = 1,303 s and the
process exits at 5.92 GB, with single-second jumps of up to +2.23 GB from the
ASEC HDF5 loads. A slope of 16 MiB per node cannot be separated from that. The
trace therefore corroborates the attribution's §9 — retention is ~0.3 GiB at
1/1000, not the 1.93 GiB a peak-RSS subtraction suggested — and the slope has to
be read either from the dedicated bench (which §9 did) or at a larger fraction.

One correction to the arithmetic §9 implies, from reading the code rather than
running anything: `population_ops.patch` deep-copies every entity table
(`_copied_tables`, `population.py:2681`), so the runner's own replay set could in
principle double the retention — but it does not, because the replay loop assigns
`expected[node_id] = current[version] = population` and most nodes take the
`current[version]` branch unchanged, so `expected` holds aliases of one object
per structural version plus the explicit `patch`/`base_expected`/`donor_columns`
entries. The observer's nineteen independent pickles remain the dominant term.

## 2. The design, in one page

`docs/us-native-scale-transport.md` (421 lines) is the authority. Its shape:

**What refused, measured from committed artifacts.** Four receipts in
`survey_population_preparation` and `graph_survey_population` carried one record
per source household or per person inside a single 64 MiB-bounded `_encode`. The
1/1000 pilot's receipt survives at 1,136,063 B (sha256 `34b362d85d2f06ac…`, byte
identical across two recovered copies and that run's own
`graph-store/objects/04/0481…/payload.bin`); re-encoding it through the
repository's own canonical separators reproduces 1,136,063 exactly. It records
`supplied_households = 1587376` with 1,584 selected, giving

> **bytes(f) = 39,242 + 692.4375 × 1,587,376 × f** — 1.024 GiB at full source

and five ceilings, in the order they bind. A 1/10 build is 158,737 source
households, so #1 and #2 bind; a full build needs all five, because 524,288 is
below the source's own 1,587,376.

| # | ceiling | refusal | households | share | after |
|---|---|---|---|---|---|
| 1 | preparation `_encode` | `PAYLOAD_LIMIT` | 96,860 | **6.10%** | 6,206,000 |
| 2 | `_origins` row budget | `ORIGIN_LIMIT` | 150,916 | **9.51%** | 9,658,000 |
| 3 | allocation payload stream | `ALLOCATION_LIMIT` | 211,924 | 13.35% | 13,563,000 |
| 4 | `_plan_document` row budget | `ORIGIN_LIMIT` | 270,284 | 17.03% | 17,298,000 |
| 5 | allocation map pre-check | `ALLOCATION_LIMIT` | 524,288 | 33.03% | 33,554,432 |

**The transport: the same byte stream, bounded differently.** `_chunks` already
defines the canonical stream independently of its materialisation, and sha256 is
a streaming hash over exactly those chunks in exactly that order, so
`_sha(_encode(x)) == _digest(x)` for every document `_encode` accepts and
`_digest` keeps returning that value for the documents it refuses. So
`preparation_sha256` and `allocation_sha256` are unchanged **in derivation and in
value**, and with them every receipt field derived from them.
`_roster_segments` closes a segment before the next chunk would carry it past
`MAX_SEGMENT_BYTES` — the old whole-document ceiling, unchanged, with
`PAYLOAD_LIMIT` still guarding each segment — and bounds the total separately
under the new `ROSTER_LIMIT` (4 GiB, 3.9× a full-source receipt).
`_spill_roster` writes the segments as `<sha256>.segment` beside a `header.json`
naming the whole-stream digest, the size and the ordered segment table: the shape
`ContentStore.put_frame` already uses. The allocation payload segments in memory,
because `SurveyPopulationAllocationKernel.run` has no filesystem path and must
not write to the store.

It is honest about what it does not buy: the bytes are still materialised once,
because `KernelResult.artifacts` is a mapping of `bytes` and
`ContentStore.put_bytes` takes whole bytes. That is 1.02 GiB at full source
against a working set the attribution puts at ~26 GiB, and removing it needs a
streaming artifact channel in `microcosm-graph`.

**The seal.** `_frame_identity` was the only remaining per-cell frame walker in
the US runtime; it now assembles the exact per-column byte string — `float64`
through a fixed-offset `(n, 37)` uint8 matrix with a per-position keep mask, the
integer widths through `int.__repr__` over one `tolist`, booleans by index,
strings by encoding each distinct value once, and `object`/`category` keeping the
walk because `1`, `True` and `1.0` are equal and hash alike there while the
encoder spells them three different ways.

**The retention.** Designed, not implemented, and §5 below says why.

## 3. Pins re-derived

**No committed pin moves.** This was re-derived rather than assumed: every
contract in `graph_implementation_inventory.json` was recomputed through
`graph_implementation._dependency_contract` against the module's own
`_covered_imports`, and all ten declared stage manifests were built:

| pin | old | new | command |
|---|---|---|---|
| `graph_implementation_inventory.json` → `survey_population_preparation.py` → `imports`, `unbound_uses_sha256`, `resource_accesses_sha256` | — | **unchanged** | `graph_implementation._dependency_contract(payload, name, _covered_imports(name, inventory))` |
| `graph_implementation_inventory.json` → `graph_survey_population.py` → same three | — | **unchanged** | same |
| every other contract in the inventory | — | **unchanged** (`contracts that do not match the tree: none`) | same, over all of them |
| all ten stage manifests | — | **build** (`stage manifests built: 10`) | `graph_implementation.implementation_manifest(stage)` for each of `STAGE_DEPENDENCIES` |
| `acs_native_coverage_binding._ACCEPTED` | — | **untouched**: it pins four ACS modules, none of which this branch edits | `shasum -a 256 <module>` |

What does move is not a committed pin and not this lane's choice: each US stage's
`implementation_hash` is over its whole module roster, so editing any inventoried
module moves it and with it every node key and every store address. #935 recorded
that this is true of any change to these files including a comment, and the same
applies here. The digests this lane *computes* — frame identities, the plan
digest, the allocation instruction bytes, `preparation_sha256`'s derivation —
keep their exact values, proven byte for byte where bytes are rebuilt.

**One review surface no pin covers.** `graph_implementation._RESOURCE_CALLS`
(`:126-150`) covers reads — `open`, `read_bytes`, `read_text`, `read_csv`,
`load`, `files`, `joinpath`, `import_module`, `exec`, `eval` — so the spill's
`Path.write_bytes` is invisible to `resource_accesses_sha256` by construction and
no committed pin would notice a redirected spill. Every component of the path is
therefore checked in the code, and the checks are listed in the design note and
the PR body. Presence is decided by `lstat`, not `Path.exists()`: a broken
symlink does not exist and the first draft of this code wrote straight through
one, which its own test caught.

## 4. The ceiling is gone

### 4a. Through the repository's own encoder, on the pilot's own receipt

`experiments/native-scale-transport/ceiling_receipt.py` replicates the recovered
1/1000 preparation artifact's per-household and per-person lists to the household
count each fraction of the 1,587,376-household source implies, and runs both
transports over it. No graph runs; nothing outside the named output directory is
written. The artifact re-encodes to its own 1,136,063 bytes through the
repository's canonical separators, so the input is the shipped preimage and not a
reconstruction of it.

| fraction | households | single `_encode` | segmented transport | segments | CPU s |
|---|---|---|---|---|---|
| 1/1000 | 1,584 | accepted, 1,136,063 B | **identical bytes** | 1 | 0.1 |
| 1/100 | 15,840 | accepted, 11,014,994 B | **identical bytes** | 1 | 1.1 |
| **1/10** | **158,400** | **REFUSED `PAYLOAD_LIMIT`** | **accepted, 109,804,304 B** | 2 | 11.5 |
| **1/1** | **1,587,168** | **REFUSED `PAYLOAD_LIMIT`** | **accepted, 1,099,892,722 B = 1.024 GiB** | 17 | 112.3 |

`_sha(payload) == _digest(document)` at all four scales, so the bytes the
transport carries hash to the digest of the same document — the identity the
whole design rests on, checked at the scales that matter rather than argued. The
measured sizes are within 0.08% of the law read off the artifact
(1,099,892,722 against 1,099,053,884 predicted at full source), and building the
full-source receipt costs **112.3 CPU-s** against the **14.6 CPU-s** the single
encode spends reaching its refusal — so raising the ceiling costs about
98 CPU-s at full source, which is what "a transport-shape decision, not a
performance trade-off" means in numbers.

### 4b. Through the graph, at 1/10

Queued at the time of writing; section 7 says what it is waiting for, what it can
add and what it cannot, and gives the prior for whether it completes inside the
brief's 21,600 CPU-s.


### 4c. The content did not move, checked on the real US frame

`experiments/native-scale-transport/receipt_parity.py` reassembles this branch's
own 1/1000 preparation receipt from the roster spill's `header.json`, verifies
every segment against its recorded digest and the whole against the stream digest
(both pass), and compares it block by block with the recovered
`native19-required-20260912` artifact:

| block | this branch | pilot | identical |
|---|---|---|---|
| `origins` | 718,800 B | 718,800 B | **yes** |
| `selection` | 393,395 B | 393,395 B | **yes** |
| `frame_sha256` | 66 B | 66 B | **yes** |
| `context_sha256`, `protocol`, `request`, `request_sha256`, `release_eligible` | | | yes |
| `source_files` | 1,004 B | 1,004 B | no |
| `native` | 413 B | 413 B | no |
| `catalogues` | 726 B | 726 B | no |
| `producer` | 21,035 B | 21,135 B | no |

Every difference is named rather than left as a difference. `source_files` differs
in **one** entry — `person-income-attachment.h5`, the known pre-restoration
`5996dcdd…` against the staged `9ebc0ef2…`, which #935's own report documents as a
difference between that pilot and this staging and not a change to any code.
`native` and `catalogues` differ only in their `receipt_sha256` fields, which
embed the producer closures and that source file. `producer` differs in 9 of its
26 module digests plus 15 `microcosm.graph` module digests inside the ACS native
source closure — modules this branch does not touch, so the comparison spans every
commit between that artifact and this head.

**The two whole-roster blocks the transport carries, and the frame identity the
per-column seal computes, are byte-identical across all of it, on the real US
frame rather than on a fixture.** That is the strongest form the
identity-equality proof can take: not "the seal agrees with an oracle on a
constructed frame", which §2 also has, but "the seal produced the same 66-byte
`frame_sha256` the build produced before any of this".

## 5. Retention: what is implemented, what is not, and why

**Implemented: the CPU half.** The attribution's item 1 names one callback as
both the memory wall and 35–46% of full-source CPU, and the CPU inside it is
`survey_population_preparation._frame_identity`, a
`for value in series: digest.update(_frame_cell_encode(value)); digest.update(b"\n")`
loop over the whole frame on every node. It is now a per-column pass that
produces the same bytes, measured at **7.68× on the frame's own dtype census**
(0.5195 → 0.0677 µs/cell) with byte equality asserted per column before either
side is timed. The baseline's own sampler ledger shows what that is worth in
situ: six chains of the form
`update_cell <- _frame_identity <- _population_stamp <- …` account for
**168.5 CPU-s** of the baseline's 2,028.7, and those chains scale with
population, so the same term at 1/10 is ~16,850 CPU-s before the change and
~2,190 after it.

The two neighbours that look like the same problem are not, and are left alone:
`asec_current_money_source._series_digest` already dumps `_data`/`_mask` bytes
for numeric dtypes and caches string tokens, and
`asec_2024_native_population._frame_identity` delegates to it.

**Not implemented: the memory half. Three mechanisms and one contract block it,
and this is the report's main negative result.**

1. **The store refuses a store-backed snapshot**, three independent ways.
   `store.py:456` zeroes the values beneath a null mask (`values[mask] = 0`)
   while `survey_atomic_geography._population_stamp` deliberately folds the
   physical storage parts *under* the mask — its own docstring says so at
   `:231-237` — so a store round trip changes every stamp. `store.py:619`
   refuses `MultiIndex`, which `test_graph_executor.py:4254` builds into the
   observed frame. `store.py:486-487` refuses `CategoricalDtype`, which `:4262`
   observes. There is also no store API that takes a `Population`, and
   `DataFrame.attrs` — asserted preserved at `:4320-4321` — is not persisted by
   `_write_frame`. **So the brief's own proposed mechanism, "snapshots are
   content-addressed to the store rather than pickled in memory", is refused by
   the store as it stands.** Anything that spills a snapshot must spill the same
   pickle bytes the executor already trusts for detachment.
2. **Object identity is pinned.**
   `result.financial_population is observed[final_node]`
   (`graph_atomic_survey_financial.py:1897`) is an identity check, not equality,
   and the completion host has two more of its own. A population dropped and
   rebuilt from a spill would be a different object and fail them.
3. **Two loops need the whole `Population`, and the replay cannot move earlier.**
   `atomic.same_replayed_population(expected[n], observed[n])` at `:1796` and
   `:1948` compares field by field and byte by byte
   (`survey_population_replay.py:132`, `:179`), and the runner's replay cannot
   be built until `run_graph` has returned because it reads the manifest's
   loaded artifacts. So **nineteen detached populations must exist somewhere
   between the observation and the comparison — RAM (~290 GiB at full source) or
   disk (~285 GiB), and there is no third place** — unless the object comparison
   is replaced by a content seal. That is a change to a verification contract,
   not an optimisation, and it is question 2 in §7.

One correction to the arithmetic that would otherwise look worse:
`population_ops.patch` deep-copies every entity table (`_copied_tables`,
`population.py:2681`), which would make the runner's replay a second full set —
but it does not, because the replay loop assigns
`expected[node_id] = current[version] = population` and most nodes take the
`current[version]` branch unchanged, so `expected` holds aliases of one object
per structural version plus the explicit
`patch`/`base_expected`/`donor_columns` entries. The observer's nineteen
independent pickles remain the dominant term.

**I did not add an unused `population_retention` mode — and the history says that
is the same call the #893 reconciliation made.** A mode that skipped the
detachment would help an observer that only seals, and I checked all six
observers in this runtime: none qualifies. Every one retains the population
objects, and four of them compare those objects with `same_replayed_population`
(`graph_atomic_survey_population` and `survey_age_calibration` compare *inside*
the observer, because their expected populations exist before the run;
`graph_atomic_survey_financial` and `graph_survey_puf55` compare afterwards).

**A `run_graph(population_retention="lazy")` was in fact written, and dropped for
this reason.** Commit `6c0f24c77` "Add opt-in lazy population retention" on PR
#893's original branch, staged as layer 08 GRAPH-ATTACHMENT-METADATA, adds
`microcosm/graph/attachments.py` (210 lines: `_StoredPopulation`,
`_LazyPopulations`), 81 lines of `executor.py`, 41 of `keys.py`, 31 of
`manifest.py` and `tests/test_population_retention.py` (528 lines). The
reconciliation dropped it
(`experiments/893-reconciliation-amendments-19-20-20260912.md:192`) because "no
call site outside the graph package passes `population_retention`; its only
consumer was the branch's own `test_lazy_snapshot_metadata_integration.py`,
dropped with it", and recorded that "if it is wanted it is a self-contained later
PR on top of piece B, which it imports."

**But reading it changes what it is for, and this matters.** `attachments.py`
imports `ContentStore`, `_write_frame` and `_payload_table`, its docstring says
"Private, non-portable population attachments and execution lifetimes. These
store references never enter node records or portable manifest identity", and its
`_LazyPopulations` is a `Mapping[str, PopulationView]`. So it made the
**manifest's attached population views** — one per structural version — lazy
against the content store. The observer retains one detached `Population` per
*node*, and §5's three store refusals are precisely why the two are not
interchangeable: a `PopulationView` is a portable Frame that already goes through
`put_frame`, and a detached snapshot is not. **Restoring layer 08 would not remove
the nineteen pickles.** It is worth restoring on its own terms and it is the
closest existing shape for the keyword, but the work above is a second thing, and
a reader who reaches for layer 08 expecting it would be disappointed.

`docs/us-native-scale-transport.md` §4 specifies the mode this work needs item by
item, with the test that pins each invariant it must not break, and with the
precedent in the same package: `graph_survey_puf55` already retains "the final
output, not all 245 executor-observed snapshots" (`:638-639`), which makes the
financial runner the outlier among the six and gives the shape the answer would
take.

## 6. What `run_graph(population_retention=...)` on main would need

`docs/us-native-scale-transport.md` §4 has it in full. In brief, five things,
each established from the code rather than proposed:

1. **A mode that skips the detachment, not just the retention** — the cost is
   `pickle.dumps` + `pickle.loads` per node (`executor.py:373`), and sealing is
   read-only, so an observer that only seals needs no detached copy.
   `test_graph_executor.py:4368-4374` already pins the converse (no observer, no
   snapshot allocated) and is the shape of the new test.
2. **A declared-consumer roster**, because the detachment exists so the observer
   may *mutate* what it keeps (`test_graph_executor.py:4183-4248`). A read-only
   view would make that mutation raise — more closed, but a behaviour change to a
   documented `microcosm-graph` contract, so it belongs to a main-only change.
3. **Invocation invariants that do not move**: exactly once per reached node, in
   `compiled.order`, on cache hits and cold runs alike (`:4153`, `:4166-4167`);
   an observer's own exception still refuses the run with its own exception type
   (`:4172-4179`); `ATOMIC_OBSERVER_ROSTER` and `COMPLETION_OBSERVER_ROSTER` still
   refuse any gap or reorder.
4. **The `graph_survey_puf55` precedent**, above.
5. **A statement about `_pure_run`**, which re-stamps every retained population on
   every `checked_view()` (`graph_atomic_survey_financial.py:676-691`), so a mode
   that drops populations must say what `FINANCIAL_NODE_POPULATION_CHANGED` means
   afterwards. `_node_population_seals` already returns `()` when there are
   no observations and no completion boundary (`:502-503`), which is the existing
   shape of "nothing retained, nothing to re-stamp".

## 7. Still in flight at the time of writing

Two measurements were running when this report was written, and the report says
so rather than leaving a gap that reads like a result.

**The 1/1000 after-run** (`~/PolicyEngine/_worktrees/microcosm-native-scale-after`,
detached at `5307249b3`, pid 54488 / pgid 54488, launched after a gate confirmed
85.5 GB available) runs the same 19-node graph at the same arguments as the
baseline and then replays it against its own store with `resume="require"`,
comparing manifest key, node keys, artifact identities, receipts and every store
object's bytes between the cold run and the replay. Its cold phase had passed
1,300 CPU-s against the baseline's 2,028.69 when this section was written. Its
roster spill is already on disk and is what the parity table in section 4c was computed
from, so the content-preservation result does not depend on the run finishing.

**The 1/10 graph run** is queued behind two gates in
`~/PolicyEngine/_worktrees/microcosm-native-scale-tenth` (detached at the branch
head, waiter pid 58401): it waits for the after-run to exit, so that neither
measurement's wall clock or peak RSS is the other's contention, and then for
`vm_stat` to show more than 70 GB available. Another lane's work held 48–62 GB
during this session, which is exactly what the gate is for; the waiter refuses
after six hours rather than starting a run that would make the machine swap.

**What the 1/10 run can and cannot add.** The ceiling question is already
answered, at full source rather than at 1/10, by §4a: the repository's own encoder
refuses `PAYLOAD_LIMIT` on the pilot's own receipt scaled to 158,400 and to
1,587,168 households, and the segmented transport carries both. What the graph run
adds is whether the *rest* of the 19-node path survives at that size, and the
honest prior is that it may not inside 21,600 CPU-s. The baseline's own ledger
gives the band: six `update_cell <- _frame_identity <- _population_stamp` chains
are 168.5 CPU-s of 2,028.69 and scale with population, and the node loops are
333.3 s of which 267.99 s is the four ASEC predictor nodes; classifying the rest
of the 1,712.62 s as source-fixed gives a scaling term somewhere between 200 and
420 CPU-s at 1/1000, or 55–275 after the seal change, i.e. **7,100–29,100 CPU-s at
1/10.** The 21,600 ceiling sits inside that band, so the run is worth making and
its outcome is not predictable from here. The harness reads the roster header and
the store's object count on every ten-second flush precisely so that a run the
ceiling truncates still carries the evidence for the question it was launched to
answer.

## 8. Tests, as CI runs them — verbatim

```
$ uv run ruff check .
All checks passed!

$ uv lock --check
Resolved 125 packages in 9ms

$ uv run python tools/ci_test_groups.py --verify
verification=ok

$ python3 -I -B -S packages/microcosm-build/tests/test_ci_test_groups.py
Ran 15 tests in 0.440s

OK

$ uv run python -m pytest packages/microcosm-graph/tests
778 passed, 1 skipped, 1 warning in 72.01s (0:01:12)

$ uv run python -m pytest packages/microcosm-build/tests/test_us_survey_population_preparation.py \
    packages/microcosm-build/tests/test_us_graph_survey_population.py \
    packages/microcosm-build/tests/test_us_survey_frame_identity_encoding.py \
    packages/microcosm-build/tests/test_us_survey_population_replay.py
359 passed, 2 warnings in 306.83s (0:05:06)
```

The four touched files were run again after the last two commits; the figure
above is the run that covered the transport, the seal and the spill together.
Beyond the gates the brief names, every test file in the build shard that imports
any of the five modules on this path was run as well — 41 files found by
`grep -rl` for `survey_population_preparation`, `graph_survey_population`,
`survey_atomic_geography`, `survey_population_domains` and
`survey_catalogue_selection`:

```
$ uv run python -m pytest $(cat dependent-tests.txt)
DEPENDENT_SUMMARY_PLACEHOLDER
```

`packages/microcosm-graph/tests` is in the battery even though this branch has
zero hunks there, because CI runs it and because the seal and the transport are
called from graph kernels.

**Not run, and why:** the whole `uv run pytest` inventory. It is the `fast` lane's
job and it takes far longer than the gates the brief names; the 41-file sweep
above is the superset of it that touches this change. The engine lanes
(`--extra us --extra uk`) were not run either: nothing on this branch is
engine-gated, and no test this branch adds or touches carries `requires_us` or
`requires_uk`.

## 9. Questions for Max

**1. Which columnar format for the receipt bodies?** The transport as shipped
uses the store's own existing shape — a `header.json` payload table plus
content-addressed bodies, exactly as `ContentStore.put_frame` already writes
`.npy` files — and adds no dependency. The bodies are still canonical-JSON
segments, so the canonical byte stream, `preparation_sha256` and every node key
are unchanged.

- **(a) Keep it as shipped.** No dependency, no digest moves, and the receipt is
  still one JSON document that any reader can concatenate and parse. The bodies
  are not columnar, so a reader still parses 1.02 GiB of JSON to answer "what was
  household 412,330's original anchor".
- **(b) Arrow/parquet bodies per entity, with the JSON header keeping the
  digests.** What the brief suggested, and what makes per-household lookups and
  per-column scans cheap. It needs **no new dependency**: `pyarrow>=15` is already
  declared by `microcosm-build` (`pyproject.toml:20`), pinned at 25.0.0 in
  `uv.lock`, installed, and already named in this very stage's
  `STAGE_DEPENDENCIES` (`graph_survey_population.py:77`). What it costs is the
  canonical byte stream: it changes, so `preparation_sha256` and **every node key
  and store address** move, once, deliberately, with a re-pin. (I first wrote in
  this report that pyarrow was absent; it is not, and the correction makes (b)
  cheaper than I had it.)
- **(c) Keep (a) now and add (b) as a second, derived artifact** — the columnar
  view published alongside the canonical stream, digest-linked to it, so lookups
  are cheap and nothing moves. Twice the bytes on disk.

I shipped (a) because it answers the ceiling question with no digest movement and
no new dependency; (b) is the better end state and is a deliberate re-pin, not a
side effect of this lane.

**2. Should the retention policy be a run option or the only behaviour — and is
the object comparison negotiable?** §5 shows the real question is not the
option's shape. Nineteen detached populations must exist between the observation
and the replay comparison, in RAM or on disk, unless
`same_replayed_population(expected[n], observed[n])` becomes a comparison of
content seals.

- **(a) Keep the object comparison; spill the observations to disk.** Memory at
  full source falls from ~300 GiB to ~24 GiB; disk rises by 19 × one snapshot,
  ~285 GiB at full source, and every `checked_view()` reloads. Nothing about the
  verification changes.
- **(b) Replace the object comparison with a content seal** —
  `_population_stamp` already folds everything `same_replayed_population`
  compares except the *type* assertions, which can be made on arrival without
  retaining. Memory falls to one live population plus the declared consumers,
  ~24 GiB at full source, no disk. It is a change to a verification contract, and
  `graph_survey_puf55` has already made that call the other way
  (`:638-639`).
- **(c) A `run_graph(population_retention=...)` option, defaulting to today.**
  Honest, but it has no consumer until (a) or (b) is chosen, which is why I did
  not add it — and it is why the #893 reconciliation dropped the one that already
  existed (`6c0f24c77`, layer 08). Note that the dropped one made the *manifest's*
  attached population views lazy, not the observer's snapshots, so restoring it
  does not answer this question; §5 has the reading.

My reading is that (b) is right and that it is your call, not mine, because it
narrows what a run proves.

**3. A third question this lane raised rather than inherited.** The 19-node path
carries a second family of ceilings — row counts, not transport shapes — and
four of them bind between 1/10 and full source:
`acs_person_coverage_columns.MAX_SELECTED_ROWS` and
`asec_current_money.MAX_PERSONS` at 1,000,000,
`acs_pums.MAX_EXACT_HOUSEHOLDS`/`MAX_EXACT_PERSON_ROWS` at 1,000,000, and
`survey_observed_age.MAX_ROWS` at 2,000,000, against a full-source 1,587,376
households and ~3,471,000 stacked (~6,943,000 cloned) persons. Should the next
lane lift them as one change with one argument, or should each be lifted by the
lane that meets it? I did not touch them, because none binds at 1/10 and each
deserves its own reading.

## 10. The pull request

| | |
|---|---|
| PR | **[PolicyEngine/microcosm#945](https://github.com/PolicyEngine/microcosm/pull/945)** — draft, and it stays draft |
| Title | Lift the whole-roster receipt ceiling and vectorise the frame seal |
| Base | `native-verify-once` (PR #935's branch) |
| Head | what `git rev-parse native-scale-transport` returns; this table does not quote a sha it cannot have written |
| Mergeable | `MERGEABLE` at the head this report was written against |
| **`packages/microcosm-graph` hunks** | **zero** — so there is nothing in this PR to mark main-only. The executor changes the retention work would need are #938's neighbours, not this branch's. |
| CI | does not run on this PR by design: `.github/workflows/test.yml` triggers on `pull_request: branches: [main]`, so only #893 reaches CI. The battery in §9 is the only gate this branch has. |

Files, against the base:

| file | what |
|---|---|
| `packages/microcosm-build/src/.../survey_population_preparation.py` | the segmented transport, the spill and its path checks, the per-column seal |
| `packages/microcosm-build/src/.../graph_survey_population.py` | the allocation payload's segments and its two ceilings |
| `packages/microcosm-build/tests/test_us_survey_population_preparation.py` | the transport and spill refusals |
| `packages/microcosm-build/tests/test_us_graph_survey_population.py` | the allocation refusals and the byte-identity against the unsegmented stream |
| `packages/microcosm-build/tests/test_us_survey_frame_identity_encoding.py` | the seal's exhaustive equality proof |
| `docs/us-native-scale-transport.md` | the design authority |
| `experiments/native-scale-transport/` | the three harnesses as they ran, the three receipt tools, and their receipts |
| `pyproject.toml` | three per-file `F401` ignores, so the committed harnesses stay byte-identical to the ones that ran |
| `PROGRESS-native-scale-transport.md` | the lane journal |

## 11. What a reader should not take from this report

- **Nothing here is a build, a certification or a release artifact.** Every
  measurement JSON carries `"release_eligible": false` and a scope line saying
  so, and no gated data was touched: the runs read the staged pilot sources by
  path and the 1/10 run reads an APFS clone of them whose bytes hash identically.
- **The 1/10 source tree is this lane's own.** The staged pilot request pins
  `fraction: [1,1000]` and `_request` requires the argument to match it, so the
  1/10 run has its own `selection-request.json`. Every other file in that tree is
  a clonefile copy of the staged original, verified by sha256; the recovered
  originals were not written, linked or moved, and their link counts are
  unchanged.
- **The seal's rate figures are on invented columns**, though the byte equality
  behind them is asserted per column against the shipped walk, and the in-situ
  effect is separately visible in the baseline's and the after-run's own sampler
  ledgers.
- **Attribution inside a run is statistical.** All three measurement JSONs carry
  16 `checks` rows with `calls=0` and status
  `"not-wrapped: runtime refuses instrumented producers (PRODUCER_CHANGED); see
  sampled_cpu_ledger"` — the runtime authenticates its own producers, so wrapping
  them refuses, and the attribution is a 0.25 s stack sampler truncated to four
  frames per chain and the top 60 chains. A chain absent from a list only means
  it fell below that list's cut.
- **The two 1/1000 runs shared the machine**, and with different neighbours: the
  baseline ended at loadavg 10.75 and ran at 99% of one core, while the after-run
  ran alongside two other lanes' work. CPU-seconds come from `process_time` and
  are unaffected by that; wall-clock is not, and the report says which is which.
