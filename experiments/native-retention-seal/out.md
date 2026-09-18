# Build lane: removing the native build's memory wall by comparing content seals

Branch `native-retention-seal`, worktree
`~/PolicyEngine/_worktrees/microcosm-native-retention`, off
`origin/native-scale-transport` at `a64f7b733`.

Implements Max's 2026-09-17 decision on the transport lane's report §10
question 2: **option (b) — replace the object comparison with a content
seal** — with 1/15 now and 1/10 when the machine is quiet, JSON segment bodies
kept, and the four row-count ceilings left to the lane that owns them.

Every line number is at this branch's head unless it names another. Line
numbers move; each one this report relies on was re-derived here.

## 1. The four answers the design note owes, in one page

`docs/us-native-retention-seal.md` (454 lines) is the authority. Its answers:

**(1) What `same_replayed_population` compares today.** Forty-four rows under
the twenty-two refusal codes `survey_population_replay.py` raises, listed field
by field in §1 of the note with the code each raises and a mutation recipe for
each: eleven at the population level, thirteen at the frame level, five per
axis (applied to every table index, every columns axis and the strata index)
and fifteen per series (applied to every column, every axis array and the
strata). The admitted dtype
set is narrow, which bounds the problem: `_series` refuses `CategoricalDtype`,
`Float64Dtype` and `DatetimeTZDtype` with `UNSUPPORTED_EXTENSION_DTYPE`, so
only masked integer/boolean, `StringDtype`, `object` and plain numpy dtypes get
through.

**(2) Which of them `_population_stamp` already folds.** Most of the *content*
— version, owners, weight kinds, mass ledger, design weights, the frame
identity, every series' storage bytes. Not the unary type assertions, not
`DataFrame.flags`, only partly the exact index class, and — this is the part
that decides the design — **it folds the one directional predicate backwards.**

**(3) How each gap is preserved.** Not by patching `_population_stamp`. The
transport report's §10(b) said it "already folds everything
`same_replayed_population` compares except the *type* assertions". That is
wrong in **both** directions, and both halves are proved by scripts committed
under `experiments/native-retention-seal/`, run at this head:

* `probe_stamp_vs_comparison.py` builds a `US_SCHEMA` population, round-trips
  its frame through `ContentStore.put_frame`/`load_frame` and prints:

  ```
  expected masked _data: [11, 71, 73]
  actual   masked _data: [11, 0, 0]
  same_replayed_population: ACCEPTS
  _population_stamp equal : False eb6879921de1 0400e853a2dd
  _frame_identity  equal  : True 774ed8184472 774ed8184472
  ```

  The comparison **accepts** that pair, because `NONCANONICAL_NULL_BACKING`
  (`survey_population_replay.py:80-83`) deliberately permits the actual side's
  null backing to be canonically zeroed — which is exactly what the store
  does (`store.py:456`). The stamp refuses it. **A stamp-equality seal would
  turn every `resume="require"` replay red.** `_population_stamp`'s own
  docstring says so (`survey_atomic_geography.py:231-237`).
* `probe_frame_identity_gaps.py` finds three discriminations
  `same_replayed_frame` makes that `_frame_identity` does not:

  ```
  nan payload 0x...11 vs 0x...12   same_replayed_frame=REFUSES NATIVE_BITS          _frame_identity=same
  flags allows_duplicate_labels    same_replayed_frame=REFUSES TABLE_TYPE_OR_FLAGS  _frame_identity=same
  quiet vs signalling NaN          same_replayed_frame=REFUSES NATIVE_BITS          _frame_identity=same
  ```

  `_cell` maps **every** NaN to `None`, so `_frame_identity` spells them all
  `null`; nothing in it folds `DataFrame.flags`; and it cannot be applied to a
  non-`US_SCHEMA` frame at all.

So the seal is purpose-built and lives in `survey_population_replay.py`, beside
the comparison it replaces. **The rule it follows, and the only substitution it
makes:** every comparison of **byte strings** becomes a comparison of their
sha256; every other predicate keeps the small object it applies to — a dtype,
its class, an index class, an axis name, a `WeightKind`, a flags value — and
applies the identical `is`/`==`; and every predicate about **one** operand is
asserted when that operand's seal is built, with the same refusal code. The
record is nested tuples of digests and small objects, **O(columns) and not
O(rows)** — measured below.

Keeping the objects rather than tokenising them is what makes the exact cases
work. `np.longlong` against `np.int64` is the sharpest: equal dtypes, equal
`str()`, equal `dtype.str`, equal bytes, different dtype **class**. Today that
refuses `SERIES_DTYPE_OR_LENGTH`; a seal keyed on spellings would have missed
it; this one retains `type(dtype)` and compares it with `is`.

**(4) What is no longer proved.** Nothing about the content. Three things
change in character:

1. Byte equality becomes sha256 equality — the substitution the whole runtime
   already rests on.
2. One-sided assertions fire when the population is observed rather than when
   it is compared, with the same code.
3. When a population carries both a one-sided defect and a two-sided
   difference, **the one-sided refusal now takes precedence.** Both refuse, so
   nothing goes unrefused, but the code between them can differ. The two
   instances found are an object axis carrying a `datetime64` leaf *and* a value
   difference (comparison `AXIS`, seal `UNSUPPORTED_OBJECT`) and a non-finite
   axis name (comparison `AXIS`, seal `UNSUPPORTED_AXIS_NAME`); the second is
   pinned by
   `test_a_non_finite_axis_name_refuses_on_both_paths_under_different_codes`.

**A fourth item was here and is now gone, which is the part worth reading.**
It said that on an object-dtype axis a difference refuses under `AXIS` rather
than `OBJECT_VALUE`, and that the fold there — the store codec's bytes — is
"never weaker than `equals`, stricter on exactly those pairs". **Both halves of
that sentence were false**, and an adversarial pass over the finished seal
proved it by running both paths:

* **weaker**, because two values with equal codec bytes need not be `==` when
  one carries its own `__eq__`, so the comparison refused `AXIS` and the seal
  **accepted** — a run accepting a replay the comparison refuses;
* **stricter** in a case the sentence did not name, because pandas holds `None`
  and every NaN interchangeable there and the codec spells them apart, so the
  comparison **accepted** and the seal refused.

The fold now reproduces the equivalence classes `Index.equals` actually has,
measured on this pin, and refuses `AXIS` when a seal is built over the one case
a digest cannot represent. The consequence for this answer is that **the
object-axis code no longer moves**: `True` against `1` and `-0.0` against `0.0`
now reach `OBJECT_VALUE` on both paths, through the byte arm the comparison
itself falls through to, and a genuine value difference reaches `AXIS` on both.
Every non-object axis kind is exact, including the three `array_equivalent` is
byte-tolerant for (`float`, `complex`, `bool`) and — since the same pass — a
masked-integer axis above `2**53`, which was folding through float64.

**Checked wider than the cases that were fixed.**
`experiments/native-retention-seal/agreement_fuzz.py` walks a seeded
pseudo-random sweep over seven axis kinds, a 21-value object pool and nine
column mutations, driving every pair through both paths:

```
$ uv run python experiments/native-retention-seal/agreement_fuzz.py \
    experiments/native-retention-seal/agreement-fuzz-receipt.json 4000
pairs=3852 disagreements=0
   2842  SURVEY_POPULATION_REPLAY_AXIS
    466  None
    130  SURVEY_POPULATION_REPLAY_MASKED_STORAGE
    112  SURVEY_POPULATION_REPLAY_OBJECT_VALUE
     99  SURVEY_POPULATION_REPLAY_PRESENT_BITS
     91  SURVEY_POPULATION_REPLAY_STRING_MASK
     49  SURVEY_POPULATION_REPLAY_SERIES_DTYPE_OR_LENGTH
     46  SURVEY_POPULATION_REPLAY_NONCANONICAL_NULL_BACKING
     16  SURVEY_POPULATION_REPLAY_STRING_VALUE
      1  SURVEY_POPULATION_REPLAY_NATIVE_BITS
```

**3,852 pairs, nine refusal codes, 466 mutual acceptances, 0 disagreements.**
What it does not cover is the item-3 precedence class: its value pool holds no
`datetime64`, because that leaf refuses at seal construction and the sweep
would report every such pair rather than the pairs it is looking for.

**The brief's stop-condition, evaluated.** It said: if the honest answer to (4)
is not "nothing", stop and put it to Max as a question with options *before*
implementing. Every defect is still refused — nothing goes unrefused, and no
code is lost — so the condition was not met and the work went on. The four
items above are changes in *character*, not in what is refused.

Items 3 and 4 were found by an adversarial pass over the finished seal rather
than by writing it. Item 4 is **§11 question 3**. Item 3 — which refusal takes
precedence when a population carries both a one-sided and a two-sided defect —
has no question of its own, because both orders refuse and both codes remain
reachable; it is recorded rather than asked. If you would rather have been
asked about a change in character as well as a change in what is refused, say
so and the next lane treats it as blocking.

## 2. The discrimination battery

The brief asked for a mutation per compared field, driven through the current
comparison and then through the seal. The battery is one file rather than two,
because a second file would mean a second copy of every mutation recipe and
that is where drift comes from:
`packages/microcosm-build/tests/test_us_survey_population_replay.py` now routes
**every** mutation through an agreement driver that runs the object comparison
*and* the seal and requires the same verdict with the same code. A mutation the
seal misses fails there.

```
$ uv run python experiments/native-retention-seal/battery_receipt.py \
    experiments/native-retention-seal/battery-receipt.json
248 passed in 1.39s
comparisons=564 agreements=564 disagreements=0 codes=21
```

**564 comparisons, 564 agreements, 0 disagreements**, over twenty distinct
refusal codes and 42 pairs both paths accept. It was 116 comparisons over 20
codes and 14 acceptances before §9d's fix pass, whose 124 new cases mostly
drive the agreement driver too. The receipt is committed at
`experiments/native-retention-seal/battery-receipt.json`, and
`battery_receipt.py` rebuilds it from the battery's own rows, so those three
figures and the table below are derived rather than counted by hand. The hook
that writes the rows is off unless `MICROCOSM_BATTERY_RECEIPT` is set and
changes no assertion. The file is **248 collected items** driving **564
comparisons**, and the decomposition is computed rather than asserted:
**238 items drive at least one comparison** and **10 drive none** — the two
seal-protocol guards, `SEAL_TYPE`, the flags fold, the two `RangeIndex`
descriptor cases, `seal_identity`'s stability, the dtype head census, the
non-finite axis name (whose whole point is that the two paths reach *different*
codes, so it cannot go through the agreement driver) and
`test_the_seal_is_proportional_to_columns_and_not_to_rows`. Two earlier drafts
of this sentence were wrong — "122 tests ... six of them", then "113 items ...
133 drive none" — and both were written before the fix pass changed the file.

| verdict both paths reached | comparisons |
|---|---|
| `AXIS` | 402 |
| `OBJECT_VALUE` | 43 |
| **accepted by both** | **42** |
| `NATIVE_BITS` | 13 |
| `POPULATION_CONTEXT` | 12 |
| `FRAME_CONTEXT` | 9 |
| `NONCANONICAL_NULL_BACKING` | 7 |
| `AXIS_NAME` | 6 |
| `SERIES_DTYPE_OR_LENGTH` | 6 |
| `MASKED_STORAGE` | 5 |
| `WEIGHT_BYTES` | 4 |
| `UNSUPPORTED_AXIS_NAME` | 3 |
| `DESIGN_BYTES` | 2 |
| `STRING_VALUE` | 2 |
| `UNSUPPORTED_EXTENSION_DTYPE` | 2 |
| `POPULATION_TYPE`, `PRESENT_BITS`, `STRATA_NAME`, `STRING_MASK`, `TABLE_TYPE_OR_FLAGS`, `UNSUPPORTED_OBJECT` | 1 each |

`AXIS` and `OBJECT_VALUE` dominate because the object-axis equality-class sweep
is 121 parametrisations each crossing the members of two classes: every
cross-class pair refuses `AXIS`, and every within-class pair either is accepted
or reaches `OBJECT_VALUE` through the byte arm.

Two codes are **unreachable**, and the receipt says so rather than leaving a
gap that reads like coverage: `FRAME_TYPE`, because `Population` validates that
its frame is a `Frame`, so no `Population` can carry anything else; and
`STRING_POLICY`, because `StringDtype.__eq__` compares storage *and* `na_value`
at this pin, so `SERIES_DTYPE_OR_LENGTH` fires first — measured, not assumed.
`STRATA_NAME` is reachable only by renaming the strata after construction,
because `Frame` normalises that name to `"stratum"`.

**The 14 accepted pairs are the half that matters most.** A purpose-built
seal is much more likely to be *stricter* than the predicate it replaces than
weaker, and a stricter seal turns a green run red. They include the store round
trip of §1; the object-scalar equivalences (`np.int64(7)` for `7`,
`np.bool_(True)` for `True`, `np.float32(1.25)` for `1.25`, `np.bytes_` for
`bytes`); and `owners` insertion order, which is compared sorted.
**The `RangeIndex` case is not one of the fourteen**, and saying it was is a
mistake this report made until it was checked against the receipt:
`RangeIndex(0, 1, 1)` against `RangeIndex(0, 1, 7)` — whose descriptors differ
and whose materialised labels do not, so a seal derived from the store's index
encoding would have folded `start`/`stop`/`step` and refused it — is checked
path by path in `test_range_index_parameters_that_no_label_shows_are_accepted`
rather than through the agreement driver, so it contributes none of the 116
rows and appears nowhere in the receipt.

### 2a. What the battery found, which is the point of having one

**The agreement driver broke the seal's first draft four times**, each a real
defect:

| what the driver caught | why |
|---|---|
| a row or column reorder reported the series code, not `AXIS` | an axis needs a **value**-equality fold as well as a byte fold, because `_axis` refuses `AXIS` on `identical` and only then falls through to the byte codes |
| `RangeIndex` against `Index` at equal labels was accepted | the seal folded `type(index).__name__`; `identical` compares the class |
| a columns-axis dtype change was accepted | the axis fold did not carry the dtype |
| a `DatetimeIndex` freq difference was accepted | `_comparables` is `['name', 'freq']` there, and no buffer carries `freq` |

**A separate adversarial pass over the finished seal found five more**, and one
was in the dangerous direction. (A *second*, wider pass later found three more,
two of them in the accept/refuse directions — §9d. The table below is the first
pass's.)

| finding | direction |
|---|---|
| an **object axis** of `["r", None]` against `["r", pd.NA]`: the comparison refuses `AXIS`, the seal **accepted** | **accept-where-refuse.** `pd.Series(index.array)` is re-inferred by pandas 3 to a `str` Series of `["r", nan]`, so both sentinels vanish. The fold now goes through `np.asarray(index.array)`. |
| a bool axis with a backing byte of `2`, and a complex axis with a NaN payload | refused under `AXIS` instead of `NATIVE_BITS`: `array_equivalent` is byte-tolerant for kinds `f`, `c` **and** `b`, not float alone |
| an axis whose dtype **class** differs (`np.longlong` against `np.int64`) | refused under `AXIS` instead of `SERIES_DTYPE_OR_LENGTH`: `Index.identical` compares dtypes with `==` only |
| a structured dtype carrying an object field | refused under a code the comparison does not have: `_array_bytes_equal` returns False rather than raising, so the caller's own code is what surfaces |
| a masked backing of `""` under the mask | the canonical-null flag was a truthiness test; the predicate is `np.all(data[mask] == 0)`, which an empty string fails |

All nine are fixed and pinned. The `_comparables` values are now compared
element-wise with `==`, as `identical` does, rather than as a tuple that
short-circuits on identity — which a name whose own `__eq__` returns `False`
would otherwise slip through.

### 2b. The seal is O(columns)

`test_the_seal_is_proportional_to_columns_and_not_to_rows` seals the same frame
at 8, 8,000 and 800,000 rows and requires the three pickled records to differ by
under 128 bytes. They differ by the width of the row counts and shapes the
record names. That is the whole memory argument, asserted rather than asserted
about.

### 2c. The six codes this change adds, five of which now have tests

The battery covers the codes that *existed*. This change **adds six**, none of
them about a population's content, and it had a test for **none** of them —
the same shape of gap as §5's inherited contract defect, where nothing was red
because nothing was tested. Five are now pinned:

| added code | what it guards | pinned by |
|---|---|---|
| `FRAME_SEAL_PROTOCOL` | a record that is not a frame seal of this protocol version | `test_a_foreign_frame_seal_record_refuses_frame_seal_protocol` |
| `POPULATION_SEAL_PROTOCOL` | the same for a population seal | `test_a_foreign_population_seal_record_refuses_population_seal_protocol` |
| `SEAL_TYPE` | `seal_identity` handed something that is not a seal record | `test_seal_identity_refuses_anything_that_is_not_a_seal_record` |
| `RETAIN_EVERY_NODE_POPULATION_FLAG` | the private retention flag is not a bool, or is set for an extension run | `test_an_invalid_retention_flag_refuses_before_source_io`, `test_retaining_every_node_together_with_a_child_property_refuses` |
| `FINANCIAL_SEALED_NODE_SCOPE` | a sealed node reaching the completion-boundary or manifest stamp arm | `test_a_sealed_node_refuses_the_completion_and_manifest_stamp_arms` |
| **`ATOMIC_OBSERVER_RETENTION`** | **the observed roster is not the declared consumers, or an arrival seal's identity moved** | **nothing — see below** |

**Each of the three seal tests was verified to go red when its own guard is
reverted**, not merely to pass: deleting the `SEAL_TYPE` require, and reducing
either protocol guard to a length check, each turns exactly its own test red
and nothing else. The population test also pins two orderings — a protocol tag
both sides agree on but that is not this version still refuses, and a record
both truncated *and* different in content refuses under the protocol code
rather than under the content code the difference would earn.

**`ATOMIC_OBSERVER_RETENTION` is not pinned, and its two conjuncts are not
equally strong.** It reads

```python
require(
    set(observed) == declared_consumers
    and all(
        seal_identity(observed_seals[node_id]) == recorded
        for node_id, recorded in observed_seal_ids.items()
    ),
    "ATOMIC_OBSERVER_RETENTION",
)
```

The first conjunct is load-bearing: it catches an observer that retained the
wrong set, and it is reachable only from inside a real nineteen-node run whose
observer has been made to lie, which is why there is no cheap test for it. The
second conjunct re-derives `seal_identity` from the very record whose recorded
identity it compares against — `observed_seals[node_id]` *is* the tuple whose
`seal_identity` was stored as `recorded` — so it can only fail if a seal
record's `repr` changed between observation and this line. A seal holds digest
bytes, dtypes, index classes and `WeightKind` members, all of stable `repr`,
so **as written that conjunct cannot fire.** It is not harmful and it is not
evidence; calling it a guard would overstate it. §11 question 7 asks what to
do about it.

## 3. What the change is, in the runner

**The roster is one, two or three nodes, not three.** It is
`financial.ATTACH_NODE`, plus the property graph's attach node when there is a
property graph, plus the tax gate when the rebase is enabled — so the base
nineteen-node run measured in §7 declares exactly **one**, which
`test_the_base_run_retains_its_declared_consumers_and_seals_the_rest` asserts
(`{'survey_predictors.attach'}`, 18 sealed). Wherever this report says "three"
it means the three the code can name, not three at once.

`graph_atomic_survey_financial` names its **declared consumers** before the run
— `financial.ATTACH_NODE`, the property graph's attach node, and the tax gate
when the rebase is enabled — because
`result.financial_population is observed[final_node]` (`:1986`, under the
IDENTITY note at `:1682`; `:1897` at the base) is an identity check and a
caller holds those objects. Those three are detached with the
executor's own `_observer_snapshot`, so what a caller receives is exactly what
it receives today. Every other observation is sealed on arrival and dropped.

Both `same_replayed_population` call sites become seal comparisons.
`_node_population_stamp` takes either arm: a retained `Population` is stamped
as before; a sealed node re-derives `seal_identity` from its own retained
record. So `FINANCIAL_NODE_POPULATION_CHANGED` keeps a defined, non-vacuous
meaning on both — *something the run retained about this node's population is
not what it was at issuance* — and `test_the_base_run_retains_its_declared_
consumers_and_seals_the_rest` proves it on a real nineteen-node run by swapping
one seal record and requiring that code.

That test is the change's own receipt:

```
19 nodes; Population objects retained: {'survey_predictors.attach'}
seal records retained: 18
run.financial_population is the retained object: True
swapping one seal record refuses FINANCIAL_NODE_POPULATION_CHANGED: True
```

**One consumer was found by test rather than by reading, and it is the one
place the saving does not apply.** `graph_survey_completion_host.py:814-816`
reads the **base** run's whole per-node population roster as its own expected
populations and hands them to `atomic._states`, which reads
`population.frame.table(entity)` for every structural node. Seals cannot serve
that. So `run_atomic_survey_financial` takes a private
`_retain_every_node_population`, set only by its own recursive base call when
`child_property is not None`, and that path retains exactly what it retains
today. The completion host's two identity fences — `self.completed is
self.observed[child.VERIFY]` and the `id(population)` roster at `:543-545` —
are over its **own** observations and do not move at all.

## 4. The `microcosm-graph` half, main-only

Commit `9bef866c5`, on its own and with its own tests, adds
`run_graph(_population_observer_detach=False)`. When it is set, the observer
receives the live admitted population and `_observer_snapshot`'s
`pickle.dumps`/`pickle.loads` round trip is not run at all. Opt-in, default
unchanged, no US import, and the keyword enters no key, no receipt and no cache
record.

It is deliberately **not** called `population_retention`. That keyword existed
on #893's branch (`6c0f24c77`, layer 08) and made the *manifest's* attached
population views lazy against the content store — a different target, as
`docs/us-native-scale-transport.md` §4 says at length. Reusing the name would
invite exactly the confusion that section warns about.

Four tests, modelled on `test_absent_observer_allocates_no_snapshot`:

| test | what it pins |
|---|---|
| `test_a_seal_only_observer_allocates_no_snapshot` | `_observer_snapshot` monkeypatched to raise is never called, and the observer still sees every node in `compiled.order` |
| `test_a_seal_only_observer_moves_no_key_receipt_or_store_object` (cold and warm) | identical manifest key, identical node keys, byte-identical store objects against a no-observer run |
| `test_a_seal_only_observer_receives_the_live_population` | **the seal-only arm only.** It asserts the observer receives the executor's own table objects in the new mode. Its default-mode arm compares one run's snapshots against a *different* run's frames, so it cannot show that the default does not hand over live objects — an adversarial pass found that and it is not fixed here; §11 question 8 |
| `test_the_detach_keyword_is_a_bool_and_does_nothing_without_an_observer` | a non-bool raises `TypeError`; the keyword alone moves no key |

**The mode's one honest cost, stated in the docstring:** in this mode the
executor no longer *enforces* that an observer cannot reach execution state, it
*trusts* the caller's declaration. The default still enforces it, and
`test_mutating_and_retained_observers_cannot_change_execution_or_cache` still
pins that.

## 5. A defect this branch inherited, repaired here, and observed in the wild

`survey_population_preparation._spill_roster` gained a `Path.read_bytes` on
`native-scale-transport` at **`b6081efcb`** ("Hash a spill segment that was
already there, rather than trusting its size"). That added one
`resource_accesses` entry, which left
`graph_implementation_inventory.json`'s declared `resource_accesses_sha256`
stale, so

```
$ implementation_hash("authenticated_survey_population_v1")
REFUSES: ValueError Unclassified US dependency/resource contract:
         microcosm.build/us_runtime/survey_population_preparation.py.
```

and `SurveyPopulationCreateKernel.implementation_hash` calls exactly that. So
**no nineteen-node graph run was possible at the base branch's tip.** Bisected
by recomputing `_dependency_contract` over the file at each revision:

| revision | contract matches the tree |
|---|---|
| `5ff889814` (the branch point) | yes |
| `5307249b3` (the transport lane's measured after-run) | yes |
| `baaf4270c`, `68395c722`, `a64f7b733` | **no** |

The transport report's §4 "No committed pin moves" was recomputed before
`b6081efcb` and is stale at its own tip. Its after-run predates the commit and
is unaffected.

**Re-reproduced from scratch**, in the detached branch-point worktree, against
that tree's own sources rather than this one's — which matters, because an
editable install in another worktree will answer for the tree you think you are
measuring, so the module's `__file__` was asserted first:

```
$ cd ~/PolicyEngine/_worktrees/microcosm-retention-branchpoint   # a64f7b733, clean
$ PYTHONPATH="$(ls -d $PWD/packages/*/src | tr '\n' ':')" python -c ...
module resolves to: .../microcosm-retention-branchpoint/packages/microcosm-build/
                    src/microcosm/build/us_runtime/graph_implementation.py
REFUSES at the base: ValueError Unclassified US dependency/resource contract:
                     microcosm.build/us_runtime/survey_population_preparation.py.
```

At this head the same call accepts, for all ten stages:
`contracts accepted at the working tree: True` in §7c's regenerated receipt.

**Observed in the wild while this lane was running.** That branch's own 1/10
run had been queued at a 70 GB memory gate since 20:39Z. The gate opened at
**22:58:25Z**, the run started, and it stopped 573 CPU-s later at 8.17 GB with

```
"status": "STOPPED_SurveyPopulationPreparationError: PREPARATION_ISSUANCE_REFUSED"
```

Its snapshots directory holds the ACS housing capture and **no** `preparation/`
roster spill, so it refused before the roster transport, inside source
authentication. `PREPARATION_ISSUANCE_REFUSED` is a catch-all that discards the
cause (`raise ... from None`, `survey_population_preparation.py:2084`), so the
artifact cannot name what refused — §8a is this lane's answer to that.

**The repair, and the guard that should have existed.** One field re-pinned:

```
resource_accesses_sha256
  old ccfed1c1acff5a1c50b538424dc129930f69c374d9841e233b3fb593a440c3aa
  new 0071f934801d15c14e4be66af12ff73daee5f65c89f61d655a23e462afdcd061
```

After it: `contracts that do not match the tree: 0` over all 124, and
`stage manifests built: 10` over all ten. The repository had **no test over the
whole inventory** — the one caller (`test_us_asec_prepared_resources.py`) builds
three of the ten stages — which is why nothing was red.
`packages/microcosm-build/tests/test_us_implementation_inventory_contracts.py`
builds all ten and checks all 124 contracts; reverting the re-pin turns both
arms red, which was verified before committing.

This is another lane's defect. It is repaired here in its own commit
(`43fb39270`) because no run this brief requires was possible without it, and
it is offered as a cherry-pick to `native-scale-transport` rather than claimed
as this lane's work.

## 6. What moves, measured rather than asserted

`implementation_manifest` folds `sha256` of the **whole file** of every module
in a stage's roster (`graph_implementation.py:431-435`), and that manifest
reaches `params["implementation"]` and therefore `node_key`.
`experiments/native-retention-seal/implementation_identity_receipt.py`
recomputes every roster file's digest at three revisions:

| comparison | roster module digests that move |
|---|---|
| `a64f7b733` → the seal and its battery (`1b0b915b0`) | **none** |
| `a64f7b733` → this lane's head | `microcosm.graph/executor.py`, in **all ten** stages |

`survey_population_replay.py`, `graph_atomic_survey_financial.py` and
`survey_atomic_geography.py` are in **no** stage roster. `executor.py` is in
**every** one. The inventory re-pin of §5 also moves `inventory_sha256`, which
is in every stage manifest.

**Said precisely, because the loose version of this sentence is wrong.** Two
things move, and they come from exactly one commit each — verifiable with
`git log a64f7b733..HEAD -- <path>`, which returns a single commit for both:

| what moves | how far it reaches | the one commit | which half |
|---|---|---|---|
| `microcosm.graph/executor.py` | the roster of **all ten** stages | `9bef866c5` | the graph half |
| `graph_implementation_inventory.json` → `inventory_sha256` | **all ten** stage manifests | `43fb39270` | the **US** half |

So **the seal and the retention change move no key**: the receipt's
`base → seal_and_battery` row — the seal plus its battery, the lane's first five
commits — is empty, and `survey_population_replay.py`,
`graph_atomic_survey_financial.py` and `survey_atomic_geography.py` are in no
stage roster at all. But **the US half is not key-neutral**, because it carries
the inherited-contract re-pin of §5, and `inventory_sha256` is in every stage
manifest. An earlier draft of this section said "the US half of this change
moves no key at all"; that was true of the seal and false of the half, and the
next paragraph of the same section already contradicted it.

The graph half moves every node key and every store address for the reason the
base branch's own report gave: a US stage's implementation hash is over its
whole module roster. That is why the executor change is its own commit and its
own future PR.

## 7. The 1/1000 cold run and its required replay

pid 82878 / pgid 82878, session leader, in the throwaway detached worktree
`~/PolicyEngine/_worktrees/microcosm-retention-after` at
`ad77fe3874d2cc5127bad7caf2f6a9cb05ed3ccb`, `RLIMIT_CPU` hard 9,060 s, harness
soft 9,000 CPU-s / 12,000 wall-s / 48 GiB, launched after the gate showed
**80.7 GB** available. The harness is a **byte-identical copy** of the base
branch's committed `harness19_after_with_required_replay.py` (sha256
`b020bec601b9aea7f99eb9675e7cbd879d48a7fad850e974aa2b9f34beb7d5da`), copied
rather than edited, as the brief required. Status
**`COMPLETED_NINETEEN_NODE_AND_REQUIRED_REPLAY`**.

### 7a. Before and after

```
| | baseline 5ff889814 | transport after 5307249b3 | retention seal (this lane) | retention seal vs after |
|---|---|---|---|---|
| runner call CPU s | 2,026.78 | 1,907.58 | 1,714.92 | -192.65 |
| runner call wall s | 2,045.91 | 1,975.99 | 1,709.70 | -266.28 |
| financial node loop wall s (19) | 277.36 | 238.33 | 196.48 | -41.85 |
| prefix node loop wall s (9) | 55.93 | 45.49 | 39.55 | -5.94 |
| outside both node loops, wall s | 1,712.62 | 1,692.17 | 1,473.68 | -218.49 |
| required replay CPU s | — | 1,664.45 | 1,566.82 | -97.63 |
| whole-process CPU s | 2,028.69 | 3,574.86 | 3,284.70 | -290.16 |
| whole-process peak RSS GB | 13.16 | 12.42 | 17.86 | +5.43 |
```

Pasted verbatim from `experiments/native-retention-seal/before_after_table.py`,
which reads the three runs' own measurement JSONs, so **no figure here is
transcribed by hand** — including the replay row and the delta column, which
the report quoted while the generator produced neither until an adversarial
pass caught it.

**Read the CPU row.** `call_cpu_seconds` comes from `process_time` and is
load-independent; the wall rows are not, and this run shared the machine with
its own twenty-one-file pytest battery while the base branch's two runs shared
it with other lanes.

**−192.65 CPU-s on the runner call, 1.112×**, and the comparison *understates*
the change: this head also carries the base branch's own `b6081efcb`, which
**adds** work (it hashes a spill segment that was previously trusted by size).

The split says where it went. The financial node loop is −41.85 s: that is the
observer, which now takes one `_observer_snapshot` instead of nineteen and one
`_population_stamp` instead of nineteen. Outside both loops is −218.49 s: that
is the two replay comparisons and the final mutation check, which now compare
seal records rather than walking nineteen pairs of populations twice. No seal
chain appears in the run's top twelve sampled CPU chains at all; the top of
that ledger is source I/O and authentication, as it was before.

### 7b. The required replay

| | |
|---|---|
| cold run | 1,714.92 CPU-s / 1,709.70 s wall |
| required replay | 1,566.82 CPU-s / 1,570.33 s wall |
| whole process | 3,284.70 CPU-s / 3,285.13 s wall / 17.86 GB peak (ceilings 9,000 / 12,000 / 48 GiB) |
| store objects written by the cold run | **5,869** — the same count as the base branch's after-run |
| every node was a store hit on the replay | **True** |
| **manifest key identical** | **True** — `a08d5536bcc93e0e065e5756a8453c78c526673b998274523906fd77546c015f` |
| node roster identical | True |
| `content_addressed` projection identical | True, `nodes_differing: []` |
| **store bytes identical** | **True** — 0 paths added, 0 changed |
| projection sha256, cold vs warm | identical |
| preparation roster header | one segment, 1,135,963 B, `06342c2bd23279a1…` |

So: **the run and its required replay export byte-identical bytes**, which is
the requirement the brief set, met at this head.

### 7c. The manifest key is not the base branch's, and here is every reason why

`a08d5536…` against that run's `bd511d92…`. The brief asked for the same key
"if the inputs are the same", and they are not.
`experiments/native-retention-seal/implementation-identity-receipt.json`
recomputes every roster module's digest at four revisions and names exactly
three causes between `5307249b3` and this head:

| # | what changed | stages moved | whose |
|---|---|---|---|
| 1 | `microcosm.build/us_runtime/survey_population_preparation.py` | `authenticated_survey_population_v1` | the base branch's own `b6081efcb` |
| 2 | `graph_implementation_inventory.json` → `inventory_sha256` | **all ten** | this lane's re-pin, forced by cause 1 (§5) |
| 3 | `microcosm.graph/executor.py` | **all ten** | this lane's seal-only observer mode |

Any one of the three moves every node key downstream of the stage it touches;
`implementation_manifest` folds `sha256` of each roster module's **whole file**
(`graph_implementation.py:431-435`) and `inventory_sha256` besides.

The generator now emits **every ordered pair** of the four revisions, so each
claim cites a row that states it. Two earlier drafts of this paragraph got it
wrong and both are worth recording: the first cited
`transport_after → us_only`, a row that says the **opposite** of what it was
quoted for (it shows `survey_population_preparation.py` moving, because
`b6081efcb` is in `a64f7b733` and not in `5307249b3`); the second cited
`base → us_only` for "the US half moves no key", when that revision is only the
seal and its battery and excludes the US re-pin that moves `inventory_sha256`
in all ten stages. §6 now carries the exact two-commit attribution.

| row | what it says |
|---|---|
| `base → seal_and_battery` | **no roster module digest moved** — the seal and its battery move no key |
| `base → head` | 10 stages, `microcosm.graph/executor.py` only — every *module* digest that moves is the graph half's |
| `transport_after → head` | 10 stages, `survey_population_preparation.py` **and** `executor.py` — causes 1 and 3 above |
| `base → transport_after` | `survey_population_preparation.py` — `b6081efcb`, which is §5's inherited defect |
| `inventory_sha256_by_revision` | `58513b5b…` at base, `transport_after` **and** `seal_and_battery`; `2f98c788…` at head — cause 2, and the one thing the US half moves |

The revision is labelled `seal_and_battery` and not `us_only`, which is what it
was called until this was checked: `1b0b915b0` is the lane's first five commits
— journal, probes, note, seal, battery — and it **precedes** both the executor
commit and the re-pin, so it is not the US half. There is no single revision
that is "every US commit and not the executor one", because `9bef866c5` lands
between them; the two-commit attribution in §6 is the exact statement instead.

The regeneration also re-checks the contract arm at the working tree at this
head: **`contracts accepted at the working tree: True`** over all ten stages,
so the test files this lane has added since the runs moved no digest and broke
no contract.

**Regenerated once more at the finished head**, after §9d's fixes changed both
`survey_population_replay.py` and `executor.py`, and the conclusion is
unchanged where it matters: `base → seal_and_battery` is still empty,
`base → head` is still `microcosm.graph/executor.py` **alone**, and all ten
contracts are still accepted. `executor.py`'s digest itself moved again
(`feb26666…` → `f9b8eca9…`), because the docstring that withdraws both
guarantees is part of the file the manifest folds — which is the same movement
already accounted for, not a new one. That the seal's own rewrite moved
**nothing** is the direct re-verification of this section's claim at the head
that ships.

### 7d. Peak RSS went up, and this report is not going to spin it

17.86 GB whole-process against the base branch's after-run's 12.42 GB. Split by
phase from the two runs' own RSS traces:

| | cold peak | replay peak |
|---|---|---|
| baseline | 13.16 GB | — |
| transport after | 12.42 GB | 11.71 GB |
| **this lane** | **13.49 GB** | **17.83 GB** |

**I cannot attribute the replay-phase difference**, and I am not going to
invent a mechanism for it. What can be said:

* The retention this change removes is, at 1/1000, **about 0.31 GB** —
  nineteen snapshots at the cost attribution's measured 9.6–9.8 bytes per cell
  over this frame's 1.70e6 cells, which is 310–317 MB, i.e. 0.29–0.30 **GiB**.
  The base branch's report and earlier drafts of this one wrote that as
  "0.31 GiB"; the arithmetic gives 0.31 GB and the two units differ by 7 %
  here. Neither the cell count nor the bytes-per-cell figure is this lane's:
  both are the cost-attribution report's, carried forward and labelled as
  such. That is an order of magnitude below the
  difference, so this comparison is not evidence about retention in either
  direction.
* `ru_maxrss` is a high-water mark of resident pages and depends on when the
  allocator and the OS return them. This run had 67–80 GB free throughout; the
  base branch's runs shared a machine where another lane held 43–62 GB. Pages
  are returned under pressure and retained without it. That is a plausible
  reading and it is **not** a measurement.
* The seal's own transient allocation is bounded by one column — it calls
  `.tobytes()` per column rather than per frame — which is the same order as
  the comparison it replaces, and no seal chain reaches the sampled ledger's
  top twelve.

What would settle it is a repeat on a quiet machine, or the same comparison at
a fraction where nineteen snapshots are not noise. §11 question 5 asks for it.

### 7e. The RSS slope the brief asked for, in the only form the artifacts support

**The harness records each node's duration and not its absolute start**
(`node_wall_times` is nineteen `[name, seconds]` pairs), so nodes cannot be
aligned to the one-second RSS series and **a true per-node slope cannot be
fitted** from the committed artifacts. Saying so is part of the answer. What
they do support is the trajectory across the runner call, which brackets the
whole node loop, from
`experiments/native-retention-seal/rss_slope_table.py`:

| | entry GB | return GB | growth GB | growth/node MB | in-call peak GB | `ru_maxrss` GB |
|---|---|---|---|---|---|---|
| baseline `5ff889814` | 0.40 | 5.91 | 5.51 | 290 | 13.16 | 13.16 |
| transport after `5307249b3` | 0.40 | 6.45 | 6.05 | 318 | 12.42 | 12.42 |
| **retention seal (this lane)** | 0.40 | **8.69** | **8.29** | **436** | **13.49** | 17.86 |

`growth/node` is the growth across the call divided by nineteen — **an average
over the call, not a fitted slope**, and not all of it is node work.

**This points against the change, and that is the finding.** At the same point
in the run — the runner call's return, before the replay — this head holds
**2.24 GB more** than the transport after-run and 2.78 GB more than the
baseline, where what the change removes is about **0.31 GB** of retention at
1/1000. The in-call peak moves the same way, 13.49 against 12.42. I am not
going to invent a mechanism for it; §7d sets out what can and cannot be said,
and the one candidate there — that this run had 67–80 GB free throughout where
the base branch's runs shared a machine holding 43–62 GB, and pages are
returned under pressure and retained without it — is a plausible reading and
**not** a measurement.

**What this does to the memory argument.** It removes the 1/1000 run as
evidence for it, in either direction: the retention removed is an order of
magnitude below the difference, so nothing here confirms or refutes the saving.
The memory argument rests on two things that are measured rather than inferred:

1. **The record's size.** `test_the_seal_is_proportional_to_columns_and_not_to_rows`
   seals the same frame at 8, 8,000 and 800,000 rows and requires the three
   pickled records to differ by under 128 bytes. A seal does not grow with rows;
   a population does.
2. **The count of retained objects, on a real nineteen-node run.**
   `test_the_base_run_retains_its_declared_consumers_and_seals_the_rest` counts
   them: **19 `Population` objects before, 1 after**, 18 sealed, and
   `run.financial_population` still the retained object. That is an object-count
   fact about the shipped code, independent of what any allocator does with the
   pages.

The fraction where the saving would show in RSS is above the ceiling of §8, so
it cannot be run today. §11 question 5 asks whether to chase the +2.24 GB on a
quiet machine first.

## 8. The 1/15 run, and the ceiling that stops it

**The base branch's own 1/10 run failed while this lane was working, and
chasing why produced this lane's largest finding.** That run had been queued at
a 70 GB memory gate since 20:39Z on 2026-09-17. The gate opened at 22:58:25Z,
the run started, and 573 CPU-s later it stopped at 8.17 GB with

```
"status": "STOPPED_SurveyPopulationPreparationError: PREPARATION_ISSUANCE_REFUSED"
```

Its snapshots directory holds the ACS housing capture and no `preparation/`
roster spill, so it refused inside source authentication, before the transport
this lane's base branch rewrote.

### 8a. Three nested catch-alls, and how the cause was recovered anyway

`PREPARATION_ISSUANCE_REFUSED` is `raise ... from None`
(`survey_population_preparation.py:2084`), so the artifact cannot name its
cause. Three rounds, each recorded in
`experiments/native-retention-seal/RUN-RECORD-diagnostic.txt`:

1. **Patch that one catch-all to carry its cause** (throwaway tree, never
   committed to the branch). Result: `ACSNativeCoverageBindingError:
   NATIVE_ISSUANCE_REFUSED` — *another* catch-all.
2. **Patch the others too.** Refused in **3 CPU-s** with
   `ACSNativeCoverageBindingError: UNREVIEWED_PREPARATION`, because
   `acs_native_coverage_binding._producer()` pins the sha256 of four named ACS
   module files (`_ACCEPTED`, `:232-237`). **Those modules cannot be
   instrumented at all**, which is itself worth recording.
3. **Observe the raise instead of changing the code.** `harness19_diag.py` is
   the tenth harness plus one block that installs a `sys.monitoring` `RAISE`
   callback. It changes no byte of the measured tree, so no source pin moves
   and no producer refuses. Its trace is committed at
   `experiments/native-retention-seal/diagnostic-1-10-raises.json`.

### 8b. What refused

```
ACSCoverageAuthenticationError: CANONICAL_SIZE
  acs_person_coverage_authentication.py :: _require
  acs_person_coverage_authentication.py :: _json.<locals>.charge
  acs_person_coverage_authentication.py :: _json.<locals>.visit
  acs_person_coverage_authentication.py :: _json.<locals>.visit
  acs_person_coverage_authentication.py :: _json
  acs_native_coverage_binding.py        :: issue_acs_native_coverage
```

**Exactly two `visit` frames**, which identifies the call: a flat list, not the
deeply nested evidence receipt further down the same function. That is the
guard at `acs_native_coverage_binding.py:563-565`:

```python
coverage._json(serialnos, min(MAX_EVIDENCE_BYTES, housing.ACS_HU_RECEIPT_MAX_BYTES))
```

`MAX_EVIDENCE_BYTES` is 2 MiB and `ACS_HU_RECEIPT_MAX_BYTES` is **1 MiB**, so
the bound on the selected-ACS-`SERIALNO` list is 1 MiB — and
`survey_population_preparation.py:1959-1961` always passes
`serialnos=acs_keys`, never `None`.

### 8c. Where it binds, computed from measured inputs

`experiments/native-retention-seal/acs_serialno_ceiling_receipt.py` reads the
`SERIALNO` width off the staged ACS source (uniformly 13 characters over
20,000 rows, so 16 canonical-JSON bytes per entry) and the ACS share of a
selection off the recovered 1/1000 pilot's own committed preparation receipt
(1,529 ACS of 1,584 selected, 96.53%):

```
1 MiB admits            : 65,535 ACS serialnos
=> refuses above ~67,892 selected households (4.28% of source)
   against the transport lane's lifted ceiling #1 of 96,860 (6.10%)

    1/1000:      1,587 selected,      1,532 ACS ->       24,516 B  fits
      1/30:     52,913 selected,     51,075 ACS ->      817,205 B  fits
      1/24:     66,141 selected,     63,844 ACS ->    1,021,506 B  fits
      1/20:     79,369 selected,     76,613 ACS ->    1,225,807 B  REFUSES
      1/15:    105,825 selected,    102,151 ACS ->    1,634,409 B  REFUSES
      1/10:    158,738 selected,    153,226 ACS ->    2,451,614 B  REFUSES
       1/1:  1,587,376 selected,  1,532,259 ACS ->   24,516,140 B  REFUSES
```

**So the binding ceiling on this path is lower than the one the base branch
lifted, and was in no census.** That lane's §5 enumerated row-count ceilings
and its §2e the transport ones; this is a canonical-JSON byte budget in the ACS
coverage binding, and it refuses every run above about 1/24. Its report's
headline is true of the ceilings it measured and does not hold for the path:
**no fraction both clears this one and exercises the 96,860 the lane lifted.**

### 8d. The 1/15 run this brief asked for: what it did

The arithmetic above predicted, **before the run and committed first**, that
1/15 refuses at about 1.63 MB against the 1 MiB cap. The run was armed, gated
and queued behind the 1/1000 anyway, because a prediction that is not run is
not a measurement, and because a refusal brackets the ceiling from the other
side.

It ran, and it refused.

| | 1/15 run |
|---|---|
| launched | 2026-09-18T00:35:15Z, gate open at **76.2 GB** available > 45 GB |
| pid / pgid | 45669 / 45669, session leader |
| source tree | `~/PolicyEngine/_worktrees/microcosm-retention-fifteenth`, detached at `0bc160099` |
| **status** | **`STOPPED_SurveyPopulationPreparationError: PREPARATION_ISSUANCE_REFUSED`** |
| CPU s | **605.65** of a 21,600 ceiling (2.8%) |
| wall s | 606.55 of 43,200 |
| **peak RSS** | **8.28 GB** of the 48 GiB ceiling (16%) |
| **nodes run** | **0** — `node_wall_times` is the empty list |
| runner call | entered at wall 2.028 s, returned at 605.690 s |
| ceilings, as set | 21,600 CPU-s / 43,200 wall-s / 48 GiB — none was lowered |
| artifact | `experiments/native-retention-seal/measurement-fifteenth-refusal.json` |

**The brief asked for the RSS slope across nodes and the runner-versus-node-loop
split. Neither exists for this run, and not because they were not recorded.**
The refusal happens inside source authentication, before the first node
executes: `node_wall_times` is empty, and the whole 605.65 CPU-s sits between
the runner call's entry and its return. The RSS series has 574 one-second
samples, whose own peak is 6.82 GB against the 8.28 GB `ru_maxrss` high-water
mark the table reports — two different measures, and the harness's own
`rss_series_note` says so — but every sample is inside source admission, so
there is no per-node slope to fit. **A row-count ceiling did not fire; a byte-budget
ceiling did** — §8e names it from the run's own raises, and §8c computes where
it binds.

**What this run does and does not establish.** It does not measure the seal at
1/15, because the seal never ran: no node executed, so no population was
observed, sealed or compared. What it establishes is the ceiling, from the
other side of the bracket: 1/24 fits and 1/15 refuses, and the refusal is the
predicted one. The seal's measurement is the 1/1000 run of §7, and the seal's
memory argument is `test_the_seal_is_proportional_to_columns_and_not_to_rows`,
neither of which this refusal touches.

### 8e. The 1/15 refusal, named from the run's own raises

The prediction named a code. `PREPARATION_ISSUANCE_REFUSED` cannot confirm it,
so the 1/15 run was repeated with the diagnostic harness of §8a round 3 —
`harness19_diag_fifteenth.py`, the 1/15 harness plus the committed
`sys.monitoring` `RAISE` block, inserted at the same point in the file, which
changes no byte of the measured tree and so moves no source pin and refuses no
producer. Same tree (`0bc160099`, `git status` clean), same staged 1/15 clone,
same gates. pid 47936 / pgid 47936, launched 2026-09-18T00:53:59Z at **79.9 GB**
available.

It refused the same way — `STOPPED_…PREPARATION_ISSUANCE_REFUSED`, 615.01
CPU-s, 618.00 wall-s, 8.17 GB `ru_maxrss` (6.88 GB across its own 583 samples),
**0 nodes** — and its trace
(`experiments/native-retention-seal/diagnostic-1-15-raises.json`, 208 raises
observed, the last 120 retained) ends:

```
ACSCoverageAuthenticationError: CANONICAL_SIZE
  acs_person_coverage_authentication.py :: _require
  acs_person_coverage_authentication.py :: _json.<locals>.charge
  acs_person_coverage_authentication.py :: _json.<locals>.visit
  acs_person_coverage_authentication.py :: _json.<locals>.visit
  acs_person_coverage_authentication.py :: _json
  acs_native_coverage_binding.py        :: issue_acs_native_coverage
ACSNativeCoverageBindingError: NATIVE_ISSUANCE_REFUSED
  acs_native_coverage_binding.py        :: issue_acs_native_coverage
  survey_population_preparation.py      :: prepare_authenticated_survey_population
SurveyPopulationPreparationError: PREPARATION_ISSUANCE_REFUSED
  survey_population_preparation.py      :: prepare_authenticated_survey_population
  graph_survey_population.py            :: run_authenticated_survey_population
  graph_atomic_survey_population.py     :: run_atomic_survey_population
  survey_population_preparation.py      :: verification_epoch
  graph_atomic_survey_financial.py      :: run_atomic_survey_financial
```

**The prediction on record is confirmed**, at the same guard and with the same
signature as 1/10: `CANONICAL_SIZE`, **exactly two `visit` frames** — the flat
`SERIALNO` list, not the nested evidence receipt further down the same function
— and raised from `issue_acs_native_coverage`. So the two fractions refuse at
the same place for the same reason, 1.63 MB and 2.45 MB against the same 1 MiB
cap, and §8c's arithmetic is now bracketed by two observed refusals rather than
by one plus a calculation.

**A defect in the 1/15 measurement harness's own receipt, which this run
corrected.** `harness19_fifteenth.py` was described as the tenth harness with
exactly three changes, and its *code* is: the fraction at `:604`, `RSS_CEILING`
at `:131`, the label. But two **receipt** fields were left as the tenth's, so
`measurement-fifteenth-refusal.json` reports `"sample": {"fraction": [1, 10]}`
and a `scope` sentence naming 1/10 and 158,737 households, while the run it
describes sampled 1/15. What is authoritative about the fraction is the call
site and the staged `sources/selection-request.json`, which carries
`"fraction":[1,15]`; both are cited in the run record. The measurement harness
is committed as it ran and has **not** been retrofitted — that would break
byte-identity with the run — and the diagnostic harness corrects both fields,
says so in its own docstring, and its receipt reads `[1, 15]` and "at 1/15 --
105,825 source households".

## 9. Tests, as CI runs them

**The battery's "before" half, at the branch point and against branch-point
sources.** The brief asked for the current comparison's refusals to be shown
before the seal's. They already were: the file this lane extends existed at
`a64f7b733` and is green there.

```
$ cd <worktree detached at a64f7b733>
$ PYTHONPATH="$(ls -d $PWD/packages/*/src | tr '\n' ':')" \
    python -m pytest packages/microcosm-build/tests/test_us_survey_population_replay.py
59 passed in 0.26s
```

**The `PYTHONPATH` is not decoration.** Run without it, the workspace's editable
install resolves `microcosm.*` to *this lane's* sources and the run silently
measures the head it was meant to compare against — checked by printing
`module.__file__`, which is how the first attempt was caught. Fifty-nine tests
at the branch point; **122** at this head, with every mutation driven through
both paths and the added codes of §2c pinned besides.

```
$ uv run ruff check .
All checks passed!

$ uv lock --check
Resolved 125 packages in 5ms

$ git diff --name-only a64f7b733...HEAD | grep '\.py$' | tr '\n' '\0' \
    | xargs -0 uv run ruff format --check
18 files already formatted

$ uv run python tools/ci_test_groups.py --verify
verification=ok

$ python3 -I -B -S packages/microcosm-build/tests/test_ci_test_groups.py
Ran 15 tests in 0.375s

OK

$ uv run python -m pytest packages/microcosm-graph/tests
783 passed, 1 skipped in 39.82s

$ uv run python -m pytest \
    packages/microcosm-build/tests/test_us_implementation_inventory_contracts.py
135 passed in 0.79s

$ uv run python experiments/native-retention-seal/battery_receipt.py \
    experiments/native-retention-seal/battery-receipt.json
122 passed in 0.84s
comparisons=116 agreements=116 disagreements=0 codes=21
```

`packages/microcosm-graph/tests` is 783 against the base branch's 778: the five
are this lane's, and every other test in that package is unchanged and green.

### 9a. The twenty-one dependent files, one pytest process each

`experiments/native-retention-seal/dependent_test_battery.py` selects every
build-shard test file naming `survey_population_replay`,
`graph_atomic_survey_financial` or `graph_implementation` — 21 of the 460 — and
runs each in its own pytest process, as CI's groups do, because one file's
`monkeypatch.setattr` on a module-level function otherwise trips
`survey_population_preparation._producer`'s live-code seal for every later
file. Three processes at a time, 2 h 20 m wall.

**Three files were red, and each for a different reason. One is this change's
defect, one is this lane's own measurement error, and one is neither.**

| file | result |
|---|---|
| `test_us_spine_blindness.py` | **1 failed**, 502 passed — **this change's defect**, fixed below |
| `test_us_graph_atomic_completion_host.py` | 1 failed, 8 passed, **23 errors** — invalidated, see below |
| `test_us_graph_atomic_person_status.py` | 11 passed, **2 errors** — invalidated, see below |
| `test_us_implementation_inventory_contracts.py` | 135 passed |
| `test_us_graph_atomic_survey_financial.py` | 8 passed (13 at this head, with the new guard tests) |
| `test_us_survey_population_replay.py` | 122 passed (246 at this head) |
| `test_us_graph_atomic_property_financial.py` | 9 passed |
| `test_us_graph_atomic_property_tax_financial.py` | 13 passed |
| `test_us_graph_atomic_survey_population.py` | 7 passed |
| `test_us_graph_us_survey_enrichment.py` | 9 passed (24 m) |
| `test_us_graph_full_puf_enrichment.py` | 57 passed |
| `test_us_property_completion_graph.py` | 27 passed |
| `test_us_survey_financial_successor.py` | 8 passed |
| `test_us_survey_origin_budget_atomic_geography.py` | 10 passed |
| `test_us_survey_age_calibration_atomic_geography.py` | 1 passed |
| `test_us_completion_support_custody.py` | 50 passed |
| `test_us_current_survey_puf_host.py` | 6 passed |
| `test_us_current_survey_puf_transfer.py` | 1 passed |
| `test_us_full_puf_output_profiles.py` | 57 passed |
| `test_us_graph_puf55_survey_ss.py` | 3 passed |
| `test_us_asec_prepared_resources.py` | 8 passed |

**The spine-blindness failure is real, and it is this change's.**
`test_runtime_population_operators_are_source_spine_blind` is a static
analyser over a reviewed roster of US runtime population operator modules, and
`survey_population_replay.py` has been on that roster since before this branch
— the branch does not touch the test. It refuses a subscript it cannot resolve
statically on a name it has inferred to be a column container, fail-closed,
because such a subscript could be reading a source-spine column. Every seal
record in this module is a **positional tuple**, compared with `expected[0]`,
`actual[6]`, `left[4]` and so on, so the analyser reported **48 offending
sites** in one module:

```
AssertionError: US runtime population operators must be source-spine blind.
Found: {'survey_population_replay.py': ('line 269:9: subscript with an
unresolvable dynamic selector (fail-closed)', ... 47 more ...,
'line 532:15: getattr with an unresolvable dynamic attribute (fail-closed)')}
```

It is a false positive about intent — a seal record is a tuple, not a column
container — and a true report about form. **Checked both ways before fixing
anything:** the same test passes at `a64f7b733` against that tree's own
sources (`1 passed in 223.35s`), and fails at this head, so the seal
introduced it. **A PR left in that state would be red in CI**, since
`test_us_spine_blindness.py` is a `test_us_*` file and this PR sets both the
`us` and `shared` path filters.

The fix is to stop indexing the records and name their fields, by unpacking:
`expected_dtype, expected_shape, expected_digest = expected` instead of
`expected[0]`, `expected[1]`, `expected[2]`. No record changed shape, order or
`repr`, so **no seal identity moves**; the module is in no stage roster and in
none of the inventory's 124 contracts, so no pin moves either. The one dynamic
`getattr(index, name, None)` over `type(index)._comparables` became a literal
reader per comparable, which is also **strictly more closed**: an index class
declaring a comparable this module has never seen now refuses `AXIS` instead of
folding `None` for it in silence.

**The completion-host and person-status results are invalidated, by me.** Both
errored, and the person-status teardown names the cause:
`ValueError: CURRENT_SURVEY_PREDICTOR_FINANCIAL_RUN_IMPLEMENTATIONS_CHANGED`,
with `SurveyPopulationGraphError: MANIFEST_NODE_STATE` at setup. Those runs
overlapped a window in which **I was editing
`survey_population_replay.py` in the working tree** — mutating the guards to
prove three new tests go red, and restoring them — and an implementation hash
is over the module file's bytes. A run whose source changes under it reports
exactly that. The failure is my measurement error, not evidence about the
change, and the honest statement is that **these two files' results are
unknown until they are re-run against a still tree.** That re-run is below.

### 9d. The adversarial verification pass, and what it found

Six independent review dimensions over the finished head — accept-where-refuse,
retention-roster completeness, executor invariants, document accuracy, battery
completeness, refusal-code preservation — with **every** finding then put to
two independent adversarial verifiers whose default answer is REFUTED, one
judging mechanism and one judging consequence. 68 agents, 0 errors,
**31 findings, 25 survived**. The receipt, with every finding's claim, both
verdicts and what was done about it, is committed at
`experiments/native-retention-seal/adversarial-verification-receipt.json`.

| disposition | findings |
|---|---|
| **fixed in code** | 5 |
| fixed in a test | 2 |
| fixed in a test **and** the documents | 2 |
| fixed in the documents | 16 |
| left open, with a question | 0 |
| **refuted by both verifiers** | 6 |

**The five code findings are the ones that matter, and three of them were
accept-where-refuse or refuse-where-accept.** Each was reproduced before being
believed and again after being fixed:

| finding | direction | now |
|---|---|---|
| an object-axis value carrying its own `__eq__` | comparison **refuses**, seal **accepted** | refuses on both, at seal construction |
| `None` against `float("nan")` on an object axis | comparison **accepts**, seal **refused** | accepted by both |
| a masked-integer axis above `2**53` | both refuse, the seal's fold was **lossy** and the code moved | exact, and `AXIS` on both |
| `run_graph`'s docstring withdrew one guarantee of two | documentation | both withdrawn by name, with the blast radius |
| the table generator produced neither the replay row nor the delta column | documentation of method | both derived |

**The sixteen documentation findings are not cosmetic.** Four of them are the
class Max's standing rule is written against — a document stating a mechanism
that is not in the code. The design note described a **dtype token** of
strings and booleans and a `_comparables` digest with a `repr()` fallback, and
built its whole "residual risk" paragraph on the fallback; the code does the
opposite of tokenising, and the note's own §4(1) argued that a token would be
wrong. Both sections now describe what ships, and §6 states the three residual
risks that are real. §6 of *this* report concluded "the US half moves no key at
all" from a revision that excludes the lane's own re-pin — the US half moves
every node key, through `inventory_sha256`.

**Two tests were vacuous and are not any more**, both found by the pass and
both verified to go red against the defect they now pin: an executor test whose
default-mode arm compared one run's objects against another run's, and the
dtype census, which asserted the same predicate twice and called no seal
function.

**What was refuted is recorded too**, because a review that only reports hits
cannot be calibrated: six findings did not survive, including a claim that the
module raises 25 codes rather than 22 (it raises 22 pre-existing plus three the
change adds, which §2c states), and three claims about battery cases that
reach their refusal for the wrong reason, which the verifiers showed reach it
for the right one.

### 9b. What this session re-derived rather than trusted

A report's numbers are worth what their re-derivation is worth, so every
figure below was recomputed at this head from the committed artifacts, and the
ones that did not match were fixed rather than explained.

| claim | how it was checked | result |
|---|---|---|
| §7a's whole before/after table | re-ran `before_after_table.py` | every cell identical |
| §7b's replay proof | read `required_replay` out of the measurement JSON | key, roster, projection, store bytes, 5,869 objects — all as quoted |
| §7d's phase peaks | split each run's `rss_series` at its replay boundary | 13.16 / 12.42–11.71 / 13.49–17.83, as quoted |
| §2's battery table | recomputed from `battery-receipt.json` | all 21 rows and the 116 total match |
| the battery receipt itself | wrote `battery_receipt.py` and regenerated it | every tally identical; rows differ only in order |
| §5's re-pin values | `git diff` of the inventory JSON | the only changed field, old and new as quoted |
| §5's headline refusal | re-ran the manifest call in the branch-point worktree, `__file__` asserted | refuses with exactly the quoted error |
| §6 and §7c's identity claims | regenerated the receipt with **every** ordered pair, then checked which commits the moving files belong to | two wrong drafts, both fixed: the seal moves no key, but the **US half does**, through the re-pin's `inventory_sha256` |
| the design note's 44 field rows | counted the ids in its own §1 tables | 44: 11 P, 13 F, 5 A, 15 S |
| the note's "narrow admitted dtype set" | ran ten dtypes through `_series` | the three extension dtypes refuse; the other seven are admitted |
| the 22 pre-existing refusal codes | diffed the code sets at `a64f7b733` and this head | 22 → 25, **none lost**, three added |
| §1's two probe outputs | re-ran both committed probes | byte-for-byte as quoted |
| "both runs measured this head" | `git diff` of `packages/*/src` against each measured tree | **no source file differs**; only two test files, added afterwards |
| the executor half's four tests | grepped each name; read the whole 19-line diff | all four exist; the diff is opt-in, default unchanged, no US import |
| nineteen retained before | read the base observer at `a64f7b733` | `observed[node_id] = population` for every node |
| eleven line citations | re-derived each at this head | seven were wrong and are fixed (§9c) |

### 9c. The citations that were wrong

`survey_population_replay.py` gained `import hashlib` at line 10, so every line
from there on moved by **one** against the base, and citations taken at the base
read one line early. The design note's §1 headers (four of them) and one in the
report were in that class. Two more had moved with this change's own edits and
pointed at unrelated code: the recursive base call that sets the retention flag
is `:1356-1371`, not `:1339-1357` — where the lines are the property-tax and
household-roles flag requires — and the executor's detachment is
`executor.py:373` inside `_observer_snapshot` at `:356`, with the per-node call
at `:2787-2791`, not `:2773-2774`, which is receipt-mass normalisation. The
identity check `result.financial_population is observed[final_node]` is `:1986`,
not `:1897`, and the ACS guard is `acs_native_coverage_binding.py:563-565`, not
`:562-564`.

Re-derived and found correct, so left alone: `graph_survey_completion_host.py`
`:781-794`, `:543-545`, `:572` and `:814-816`; `graph_context.py:130-143`;
`survey_atomic_geography.py:230-268` and `:231-237`; and the design note's two
seal call sites `:1882` and `:2042`.

## 10. The pull request

**[PolicyEngine/microcosm#950](https://github.com/PolicyEngine/microcosm/pull/950)**
— *Compare replayed populations by content seal instead of by retained object*.
Draft, base `native-scale-transport`, `MERGEABLE`. **It stays draft, and it was
never marked ready.** Its body carries the four answers, the fail-closed
statement, the declared-consumer roster, the inherited-defect repair, the what
moves table, and the main-only hunk table — the two
`packages/microcosm-graph/` files, whose change is commit `9bef866c5` on its
own.

## 11. Questions for Max

**1. Where should the inherited contract repair live?** §5's re-pin is another
lane's defect, repaired here because no run this brief requires was possible
without it.

- **(a) Leave it on this branch**, as shipped. This PR is against
  `native-scale-transport`, so the repair reaches that branch when this one
  merges — but that lane's own 1/10 run and anything else it starts stays
  broken until then.
- **(b) Cherry-pick `43fb39270` onto `native-scale-transport` now**, and drop
  it here on the rebase. That unblocks that lane immediately. It also moves
  `inventory_sha256`, and therefore every node key, on **that** branch rather
  than this one — which is arguably where it belongs, since `b6081efcb` is
  that lane's commit.
- **(c) (b), plus correct that lane's report §4 in place.** Its "No committed
  pin moves" table was recomputed before `b6081efcb` and is stale at its own
  tip; the journal rule in `CLAUDE.md` is to historicize a currency claim
  rather than leave it to mislead.

My reading is (c). It is your call because it edits another lane's report.

**2. The executor half is separable, and the memory wall does not need it.**
The wall is the *retention* of nineteen populations, not the *detachment*,
which is transient and one at a time. The US half alone — seal on arrival,
retain the declared consumers — removes the wall. The executor keyword removes
the remaining per-node `pickle.dumps`/`pickle.loads`, which is CPU and transient
peak, and it is what moves every node key (§6).

- **(a) Ship both, as shipped.** One re-pin, both benefits. The runner passes
  `_population_observer_detach=False` unconditionally, so the two halves are
  coupled in this branch.
- **(b) Decouple them.** The US half lands with no key movement at all; the
  executor keyword goes to `main` as its own PR and the runner starts passing
  it afterwards. Two changes, one key movement, and the wall is gone from the
  first.
- **(c) Drop the executor half from this lane.** Honest, but it throws away a
  measured CPU saving for a key movement that §6 shows is happening anyway
  through the inventory re-pin.

I shipped (a) because the brief asked for the executor mode in its own commit,
which it is, and because the re-pin moves the keys regardless. (b) is the
cleaner history if you want the US half reviewable on its own.

**3. This question is withdrawn, and what replaced it is worth your attention
more than the question was.** It used to ask whether an object-dtype axis
should keep `OBJECT_VALUE`, on the premise that the seal refuses `AXIS` where
the comparison refuses `OBJECT_VALUE` and that the fold there is "never weaker
than `equals`". The adversarial pass of §9d showed the premise was wrong in
both directions — the seal was **weaker** for a value carrying its own
`__eq__`, and **stricter** for `None` against `float("nan")` — and the fold was
rewritten to reproduce the equivalence classes `Index.equals` actually has. The
object-axis code no longer moves, so there is nothing to decide.

**What I would rather you ruled on is the process, not the code.** Two of the
three divergences existed in a change whose own report claimed, in §1, that
"every defect is still refused", and the report had a battery of 116 agreeing
comparisons behind that claim. The battery agreed because it tested the cases
its author thought of. What caught the divergences was an adversarial pass that
tried to break the claim, and what would have caught them earlier is the
3,852-pair seeded sweep that is now committed — 40 lines, seconds to run.

- **(a) Require a seeded agreement sweep beside any hand-built battery**, in
  this lane's own standard and the next one's, whenever a change replaces a
  predicate with a fold. Cheap, and it is the thing that would have worked.
- **(b) Require an adversarial pass before a lane reports.** More expensive —
  68 agents here — and it found things no sweep would, including four
  documented mechanisms that are not in the code.
- **(c) Neither as a rule; keep doing both when the change warrants it.**

My reading is **(a) always, (b) for anything that moves a fail-closed
predicate**, which this was.

**4. Who lifts the ACS serialno ceiling, and does the base branch's headline
need amending?** §8 shows a 1 MiB canonical-JSON cap on the selected-ACS-
`SERIALNO` list refusing above about 4.28% of source — **below** the 6.10%
that branch lifted, and in no census. Two separate calls:

- **The lift itself.** The brief told this lane to leave the four row-count
  ceilings to the lane that owns them. This is a fifth, of a different kind — a
  byte budget, not a row count — and this lane left it alone on the same
  principle. **(a)** the ceiling lane takes it with the other four; **(b)** it
  is its own change, because unlike the row counts it is a *transport* shape
  and belongs with the transport work that lifted its siblings; **(c)** this
  lane takes it now, which would make the 1/15 run possible but widens a lane
  that is already carrying another branch's contract repair.
- **The base branch's report.** Its §5 census and its §2e table read as "a 1/10
  build meets no ceiling this lane did not lift". That is true of the ceilings
  it enumerated and is not true of the path, which its own 1/10 run then
  demonstrated. Do you want that historicized in place, as `CLAUDE.md`'s
  journal rule asks, or left for the ceiling lane's own report to correct?

My reading: **(b)** for the lift, and yes for the amendment — the sentence is
the kind that gets quoted later.

**5. Resident memory went up at every point this run measures it, and I cannot
say why — do you want it chased?** §7d: the replay phase peaks at 17.83 GB
against the base branch's 11.71 GB. §7e: at the runner call's *return*, before
any replay, this head holds 8.69 GB against 6.45 GB, and the in-call peak is
13.49 against 12.42. The retention this change removes is about 0.31 GB, an
order of magnitude below any of those differences, so the comparison is not
evidence about retention in either direction — but three figures moving the
same way is worth a sentence better than the one I can give. Options: **(a)** repeat the 1/1000 after-run on
a quiet machine, which is about an hour and settles whether it is allocator
behaviour under no memory pressure; **(b)** leave it until a fraction where
nineteen snapshots are not noise, which needs the ceiling of §8 lifted first;
**(c)** accept it as unattributed. I would take (a), because an unexplained
+6 GB next to a change whose purpose is memory is the kind of loose end that
gets quoted against it.

There is one cheap change that would shrink the seal's transient regardless,
and this lane deliberately did **not** make it: `_array_seal` calls
`.tobytes()` per column, which copies it, where `hashlib.update` accepts the
array's buffer directly. Making it after the measurements would have meant the
committed head was not the head that was measured. It is a one-line follow-up.

**6. The completion path still retains everything, and that is where the
remaining wall is.** `graph_survey_completion_host` hands the base run's whole
population roster to `_states`, which reads each frame. Should the next lane
give the completion host the same treatment — which means finding a frame-free
form for `_states`' per-node cell census — or is the base 19-node path the only
one whose memory matters?

**7. `ATOMIC_OBSERVER_RETENTION`'s second conjunct cannot fire — keep it, test
it, or cut it?** §2c: it re-derives `seal_identity` from the very record whose
recorded identity it compares against, so it is a self-comparison as written.
The first conjunct is real and is reachable only from inside a full run.

- **(a) Leave both, as shipped.** The self-comparison costs nineteen sha256
  digests over small tuples and documents the intent. §2c says plainly that it
  cannot fire, so nobody will quote it as coverage.
- **(b) Keep it and pin the first conjunct** with a fresh nineteen-node run
  whose observer is patched to retain the wrong roster. That is a real test of
  the code that matters, at the cost of one more ~145 s test file run in the
  build shard's slowest group.
- **(c) Cut the second conjunct** and keep `set(observed) == declared_consumers`
  alone, which is the half that can fail.

My reading is **(b)**, because the first conjunct is the fence that would catch
a future edit to the declared-consumer roster, and it is the one thing in this
change with no test at all. I did not do it here because it adds a run to the
slowest test file and the brief's measurement work had the machine.

**8. One executor test's default-mode arm is vacuous — worth a fix, or worth
just saying so?** `test_a_seal_only_observer_receives_the_live_population`
asserts the new mode hands over the executor's own table objects, which it
does. Its *other* arm is meant to show the default does **not**, and it
compares one run's snapshots against a **different** run's frames, so it would
pass whatever the default did. `test_absent_observer_allocates_no_snapshot` and
`test_mutating_and_retained_observers_cannot_change_execution_or_cache` do pin
the default's behaviour, so nothing is unpinned — but that table row in §4
claimed a discrimination this test cannot make, and §4 now says so.

- **(a) Fix the test** so both arms observe the same run, which is a small
  change to a `microcosm-graph` test and belongs with the main-only PR.
- **(b) Leave it and keep §4 honest about what it pins.**

My reading is (a), in the main-only PR rather than here, so the graph half
arrives with its own tests intact.

## 12. What a reader should not take from this report

- **Nothing here is a build, a certification or a release artifact.** Every
  measurement JSON carries `"release_eligible": false` and a scope line saying
  so; no gated data was touched and none left the machine. The 1/1000 run reads
  the staged pilot sources by path, and the 1/15 run reads an APFS clone of
  them whose bytes hash identically — with the clone proved not to change the
  originals' link count (`cp -c` on this filesystem yields a distinct inode and
  leaves `st_nlink` at its prior value, tested directly; the staged inputs
  carry a pre-existing second link, uniformly, from elsewhere in the recovered
  tree).
- **The memory figures at 1/1000 are noise, not a result.** Nineteen snapshots
  at 1/1000 are about 0.31 GB against a 1–2 GB RSS sawtooth and single-second
  jumps of over 2 GB from the ASEC HDF5 loads — the transport lane's baseline
  said so and this lane's trace agrees. The retention saving has to be read at
  a larger fraction or from the record's own size, which
  `test_the_seal_is_proportional_to_columns_and_not_to_rows` gives directly.
- **CPU-seconds are `process_time` and wall-clock is not.** This lane's 1/1000
  run shared the machine with its own pytest suite and with the 1/10 refusal
  diagnostic; the transport lane's two runs shared it with other lanes. The
  report leads with the CPU figure for that reason, exactly as that lane's did.
- **The battery's frames are invented.** They are `EntitySchema`-level frames
  built in the test file, not US source frames, and the mutations are
  constructed rather than observed. What that buys is coverage of cases a real
  frame does not carry; what it does not buy is evidence that a real frame
  carries them.
- **`PREPARATION_ISSUANCE_REFUSED` is a catch-all.** Any statement here about
  *what* refused at 1/10 or at 1/15 comes from the diagnostic runs of §8a and
  §8e, not from the shipped code's own artifact. The committed traces come from
  the round that **patches nothing**: it installs a `sys.monitoring` `RAISE`
  callback and changes no byte of the measured tree. The two rounds that did
  patch the catch-alls are recorded in
  `experiments/native-retention-seal/RUN-RECORD-diagnostic.txt` for what they
  found, and the second of them refused before it started — which is itself
  §8a's third finding.
- **The inherited defect is another lane's**, and the bisect above is this
  lane's reading of that lane's commits, not an accusation about intent: the
  contract went stale because nothing in the repository tested it.
- **The measured runs are not all at one head, and §9b says which is which.**
  The 1/1000 after-run of §7 was **re-run at the finished head** after the
  fixes of §9d changed the seal; the first after-run's figures are kept beside
  it, because they are the ones the base branch's comparison was built against.
  The 1/15 run and its diagnostic predate the fixes and were not repeated: the
  refusal they report happens inside source authentication, before the first
  node, so no seal code runs in either and the fixes cannot reach it.
- **Two of the 21 dependent files have no result at this head.** The
  completion-host and person-status runs were invalidated by my own concurrent
  edits to `survey_population_replay.py` while they ran, which an
  implementation hash is over. §9a says so, and what replaced them is a clean
  re-run — not an argument that the first result did not count.
- **The agreement sweep is evidence about the pairs it draws, and nothing
  else.** 3,852 pairs over seven axis kinds, a 21-value object pool and nine
  column mutations, with a fixed seed. It found no disagreement; it did not
  prove there is none, and its pool deliberately excludes the `datetime64`
  leaf whose refusal is the precedence class of §1's item 3.
- **This report has been wrong in public more than once**, and each time the
  wrong version is left visible with what replaced it: "the US half moves no
  key at all", "never weaker than `equals`", "0.31 GiB", the section numbering,
  seven line citations, the 122-to-116 reconciliation and the fourteen
  accepted pairs. A report that only shows its final state hides how much of
  it was checked.
