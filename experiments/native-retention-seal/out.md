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
each: eleven at the population level, thirteen at the frame level, five per axis (applied to every
table index, every columns axis and the strata index) and fifteen per series
(applied to every column, every axis array and the strata). The admitted dtype
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
  (`survey_population_replay.py:79-82`) deliberately permits the actual side's
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

**(4) What is no longer proved.** Nothing about the content. Four things change
in character, and the note states all four:

1. Byte equality becomes sha256 equality — the substitution the whole runtime
   already rests on.
2. One-sided assertions fire when the population is observed rather than when
   it is compared, with the same code.
3. When a population carries both a one-sided defect and a two-sided
   difference, the one-sided refusal now takes precedence. Same codes,
   different order between them.
4. On an **object-dtype axis**, a difference refuses under `AXIS` rather than
   under `OBJECT_VALUE`, because `Index.equals` there is an element-wise `!=`
   over arbitrary Python objects — it holds `True` equal to `1` and `-0.0`
   equal to `0.0` — which no digest reproduces. The fold there is the store
   codec's bytes: never weaker than `equals`, stricter on exactly those pairs.
   Every non-object axis kind is exact, including the three `array_equivalent`
   is byte-tolerant for (`float`, `complex`, `bool`).

Every defect is still refused, so this did not go back to the owner as a
blocking question. Items 3 and 4 were found by an adversarial pass over the
finished seal, not by writing it, and are §10 question 1 below.

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
$ MICROCOSM_BATTERY_RECEIPT=... uv run python -m pytest \
    packages/microcosm-build/tests/test_us_survey_population_replay.py
119 passed in 0.99s
```

**116 comparisons, 116 agreements, 0 disagreements**, over twenty distinct
refusal codes and 14 pairs both paths accept. The receipt is committed at
`experiments/native-retention-seal/battery-receipt.json`; the hook that writes
it is off unless the environment variable is set and changes no assertion.

| verdict both paths reached | comparisons |
|---|---|
| `AXIS` | 16 |
| **accepted by both** | **14** |
| `NATIVE_BITS` | 13 |
| `POPULATION_CONTEXT` | 12 |
| `FRAME_CONTEXT` | 9 |
| `OBJECT_VALUE` | 9 |
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
`bytes`); `owners` insertion order, which is compared sorted; and
`RangeIndex(0, 1, 1)` against `RangeIndex(0, 1, 7)`, whose descriptors differ
and whose materialised labels do not — a seal derived from the store's index
encoding would have folded `start`/`stop`/`step` and refused it.

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
was in the dangerous direction:

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

## 3. What the change is, in the runner

`graph_atomic_survey_financial` names its **declared consumers** before the run
— `financial.ATTACH_NODE`, the property graph's attach node, and the tax gate
when the rebase is enabled — because
`result.financial_population is observed[final_node]` (`:1897`) is an identity
check and a caller holds those objects. Those three are detached with the
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
| `test_a_seal_only_observer_receives_the_live_population` | the observer really does receive the executor's own table objects, and the default mode really does not |
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
artifact cannot name what refused — §6 is this lane's answer to that.

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
| `a64f7b733` → this lane's **US-only** commits | **none** |
| `a64f7b733` → this lane's head | `microcosm.graph/executor.py`, in **all ten** stages |

`survey_population_replay.py`, `graph_atomic_survey_financial.py` and
`survey_atomic_geography.py` are in **no** stage roster. `executor.py` is in
**every** one. The inventory re-pin of §5 also moves `inventory_sha256`, which
is in every stage manifest.

So: **the US half of this change moves no key at all**, and the
`microcosm-graph` half moves every node key and every store address — for the
reason the base branch's own report gave, that a US stage's implementation hash
is over its whole module roster. That is why the executor change is its own
commit and its own future PR.

## 7. The 1/1000 cold run and its required replay

*(in flight when this section was written; §7 is completed below once the
replay's verdict is in)*

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
guard at `acs_native_coverage_binding.py:562-564`:

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

### 8d. What that means for the 1/15 run this brief asked for

The arithmetic above predicts the 1/15 run refuses the same way, at about
1.63 MB against the 1 MiB cap. It is armed, gated and queued behind the 1/1000
anyway, because a prediction that is not run is not a measurement, and because
a refusal brackets the ceiling from the other side.

*(the run's own outcome follows once it has run)*

## 10. Tests, as CI runs them

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
at the branch point; 119 at this head, with every mutation driven through both
paths.

```
$ uv run ruff check .
All checks passed!

$ uv lock --check
Resolved 125 packages in 4ms

$ uv run ruff format --check <every .py this branch touches>
12 files already formatted

$ uv run python tools/ci_test_groups.py --verify
verification=ok

$ python3 -I -B -S packages/microcosm-build/tests/test_ci_test_groups.py
Ran 15 tests in 0.357s

OK

$ uv run python -m pytest packages/microcosm-graph/tests
783 passed, 1 skipped in 48.89s

$ MICROCOSM_BATTERY_RECEIPT=... uv run python -m pytest \
    packages/microcosm-build/tests/test_us_survey_population_replay.py
119 passed in 0.99s
```

`packages/microcosm-graph/tests` is 783 against the base branch's 778: the five
are this lane's, and every other test in that package is unchanged and green.

### 10a. The twenty-one dependent files, one pytest process each

*(in flight)*

## 9. Questions for Max

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

**3. Should an object-dtype axis keep `OBJECT_VALUE`?** §4 item 4: on an
object axis the seal refuses under `AXIS` where the comparison refuses under
`OBJECT_VALUE`, because `Index.equals` there is an element-wise `!=` over
arbitrary Python objects that no digest reproduces.

- **(a) Keep it as shipped.** The refusal is preserved; only the code moves,
  and only for an object-dtype axis.
- **(b) Retain the axes.** An index is `O(rows)` but 8 bytes a row, not a
  frame: nineteen person indexes at 1/10 are about 0.5 GB against the 29 GB
  the populations were. Keeping the objects for axes alone would make every
  axis code exact.
- **(c) Refuse object-dtype axes outright in the seal.** Strictly more closed,
  and a behaviour change to a case no US frame is known to carry.

I shipped (a). (b) is the only one that makes the codes exact, and it is not
free.

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

**5. The completion path still retains everything, and that is where the
remaining wall is.** `graph_survey_completion_host` hands the base run's whole
population roster to `_states`, which reads each frame. Should the next lane
give the completion host the same treatment — which means finding a frame-free
form for `_states`' per-node cell census — or is the base 19-node path the only
one whose memory matters?

## 11. What a reader should not take from this report

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
  at 1/1000 are about 0.31 GiB against a 1–2 GB RSS sawtooth and single-second
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
  *what* refused at 1/10 comes from the diagnostic run of §6, which patches the
  catch-all in a throwaway tree, and not from the shipped code's own artifact.
- **The inherited defect is another lane's**, and the bisect above is this
  lane's reading of that lane's commits, not an accusation about intent: the
  contract went stale because nothing in the repository tested it.
