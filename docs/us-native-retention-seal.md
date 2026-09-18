# US native build: replacing the replayed-population object comparison with a content seal

Design authority for the change Max chose on 2026-09-17 — option (b) of the
transport lane's report §10 question 2
(`experiments/native-scale-transport/out.md`), whose own authority is
`docs/us-native-scale-transport.md` §3d and §4.

The problem, restated from that authority and re-derived here: the financial
runner compares its own replay against the executor's observation with
`atomic.same_replayed_population(expected[n], observed[n])`
(`graph_atomic_survey_financial.py:1796` and `:1948` **at the base**, now the
two `same_replayed_population_seals` calls at `:1882` and `:2042`), and the
replay cannot be
built until `run_graph` has returned. So nineteen detached `Population` objects
must exist between observation and comparison — about 290 GiB at full source —
unless the comparison becomes a comparison of content seals.

Every line number is at this branch's head unless it names another. Line
numbers move; re-derive the ones you rely on.

## 0. The report's proposed mechanism does not hold, and this is the first thing to say

Report §10 question 2(b) says:

> `_population_stamp` already folds everything `same_replayed_population`
> compares except the *type* assertions, which can be made on arrival without
> retaining.

That sentence is wrong in **both** directions. Both halves are proved by
scripts committed under `experiments/native-retention-seal/`, run at this head.

**`_population_stamp` is too strict — it would refuse a required replay.**
`probe_stamp_vs_comparison.py` builds a `US_SCHEMA` population whose
`nullable_integer` column carries `_data = [11, 71, 73]` under a mask
`[False, True, True]`, round-trips its frame through
`ContentStore.put_frame`/`load_frame` (which zeroes values beneath a null mask,
`store.py:456`), and reports:

```
expected masked _data: [11, 71, 73]
actual   masked _data: [11, 0, 0]
same_replayed_population: ACCEPTS
_population_stamp equal : False eb6879921de1 0400e853a2dd
_frame_identity  equal  : True 774ed8184472 774ed8184472
```

`same_replayed_population` **accepts** that pair, because
`NONCANONICAL_NULL_BACKING` (`survey_population_replay.py:80-83`) deliberately
permits the actual side's null backing to be canonically zeroed. The stamp
refuses it. `_population_stamp`'s own docstring says exactly this
(`survey_atomic_geography.py:231-237`): "Never persist it or compare it across
reconstructions: it folds physical storage parts (masked `_data` under nulls)".
A stamp-equality seal would turn every `resume="require"` replay red.

**`_frame_identity` is too weak — it would miss real defects.**
`probe_frame_identity_gaps.py` finds three discriminations `same_replayed_frame`
makes that `_frame_identity` does not:

```
nan payload 0x...11 vs 0x...12     same_replayed_frame=REFUSES NATIVE_BITS
_frame_identity=same
flags allows_duplicate_labels      same_replayed_frame=REFUSES
TABLE_TYPE_OR_FLAGS   _frame_identity=same
quiet vs signalling NaN            same_replayed_frame=REFUSES NATIVE_BITS
_frame_identity=same
```

`_cell` (`survey_population_preparation.py:732-745`) maps **every** NaN to
`None`, so `_frame_identity` spells all NaN payloads `null`; and nothing in it
folds `DataFrame.flags`. It is also inapplicable to a non-`US_SCHEMA` frame at
all (`FRAME_TYPE`, `:977-981`).

**So the seal is purpose-built and lives beside the comparison it replaces**, in
`survey_population_replay.py`, folding exactly the bytes each comparison
compares — no more, no less. It reuses neither existing seal.

## 1. Every field and assertion `same_replayed_population` compares today

Read off `survey_population_replay.py` at this head. **Kind** is `unary` for a
predicate about one side that can be asserted on arrival, `binary` for a
comparison between the two sides that a seal must fold, and `directional` for
the one predicate that is not symmetric.

### 1a. Population level (`:179-208`)

| # | predicate | code | kind |
|---|---|---|---|
| P1 | `type(expected) is Population`, `type(actual) is Population` | `POPULATION_TYPE` | unary, both sides |
| P2 | `type(version) is str` (both) and `expected.version == actual.version` | `POPULATION_CONTEXT` | unary + binary |
| P3 | `tuple(sorted(owners.items()))` equal — order-**insensitive** | `POPULATION_CONTEXT` | binary |
| P4 | `all(type(v) is str for v in actual.owners.values())` | `POPULATION_CONTEXT` | unary, actual only |
| P5 | `tuple(weight_kind.items())` equal — order-**sensitive** | `POPULATION_CONTEXT` | binary |
| P6 | `all(type(v) is WeightKind for v in actual.weight_kind.values())` | `POPULATION_CONTEXT` | unary, actual only |
| P7 | `type(mass_ledger) is tuple` (both) | `POPULATION_CONTEXT` | unary |
| P8 | `all(type(r) is MassRecord ...)` over both ledgers | `POPULATION_CONTEXT` | unary |
| P9 | `canonical_json([asdict(r) for r in mass_ledger])` equal | `POPULATION_CONTEXT` | binary |
| P10 | `tuple(design_weights)` equal — key **order** | `POPULATION_CONTEXT` | binary |
| P11 | per entity `_array_bytes_equal(design_weights[e], …)`: `type is np.ndarray`, dtype equal, `not hasobject`, shape equal, `tobytes()` equal | `DESIGN_BYTES` | unary + binary |

### 1b. Frame level (`:132-176`)

| # | predicate | code | kind |
|---|---|---|---|
| F1 | `isinstance(expected, Frame)`, `isinstance(actual, Frame)` | `FRAME_TYPE` | unary |
| F2 | `schema` equal | `FRAME_CONTEXT` | binary |
| F3 | `entities` equal (ordered tuple) | `FRAME_CONTEXT` | binary |
| F4 | `links == () ` on **both** | `FRAME_CONTEXT` | unary |
| F5 | `canonical_json(_encode_frame_metadata(metadata))` equal | `FRAME_CONTEXT` | binary |
| F6 | `all(type(r) is MassChangeRecord ...)` over both mass logs | `FRAME_CONTEXT` | unary |
| F7 | `canonical_json([asdict(r) for r in mass_log])` equal | `FRAME_CONTEXT` | binary |
| F8 | `weighted_entities` equal | `FRAME_CONTEXT` | binary |
| F9 | `type(table) is pd.DataFrame` (both) | `TABLE_TYPE_OR_FLAGS` | unary |
| F10 | `left.flags == right.flags` | `TABLE_TYPE_OR_FLAGS` | binary |
| F11 | strata name: `_name_bytes` equal | `STRATA_NAME` | binary |
| F12 | per weighted entity `left.kind is right.kind` | `WEIGHT_BYTES` | binary |
| F13 | per weighted entity `_array_bytes_equal(values)` | `WEIGHT_BYTES` | unary + binary |

### 1c. Axis (`:121-129`), applied to every table index, every table columns
axis, and the strata index

| # | predicate | code | kind |
|---|---|---|---|
| A1 | `type(expected) is type(actual)` | `AXIS` | binary |
| A2 | `expected.identical(actual)` — pandas: `equals()` **and** every name in `type(index)._comparables` **and** exact class **and** dtype | `AXIS` | binary |
| A3 | `not isinstance(expected, pd.MultiIndex)` | `AXIS` | unary (expected; A1 carries it to actual) |
| A4 | `_name_bytes(name)` equal — `canonical_json(_axis_name_payload(name))` | `AXIS_NAME` | binary, plus unary encodability (`UNSUPPORTED_AXIS_NAME`) |
| A5 | `_series` over the axis array | the S codes | |

### 1d. Series (`:48-111`), applied to every table column, every axis array and the strata

| # | predicate | code | kind |
|---|---|---|---|
| S1 | `type(x) is pd.Series` (both) | `SERIES_DTYPE_OR_LENGTH` | unary |
| S2 | `type(expected.dtype) is type(actual.dtype)` | `SERIES_DTYPE_OR_LENGTH` | binary |
| S3 | `expected.dtype == actual.dtype` | `SERIES_DTYPE_OR_LENGTH` | binary |
| S4 | `len(expected) == len(actual)` | `SERIES_DTYPE_OR_LENGTH` | binary |
| **masked branch** — `BooleanDtype`, or `ExtensionDtype` and integer | | | |
| S5 | `_array_bytes_equal(lm, rm)` — the null masks | `MASKED_STORAGE` | binary |
| S6 | `lm.dtype is bool`; `lm.shape == (n,)`; `type(ld) is type(rd) is np.ndarray`; `not ld.dtype.hasobject`; `ld.shape == rd.shape == lm.shape` | `MASKED_STORAGE` | unary + shape |
| S7 | `ld.dtype == rd.dtype` | `MASKED_STORAGE` | binary |
| S8 | `_array_bytes_equal(ld[~lm], rd[~rm])` — the **present** values | `PRESENT_BITS` | binary |
| S9 | `_array_bytes_equal(ld, rd) or bool(np.all(rd[rm] == 0))` | `NONCANONICAL_NULL_BACKING` | **directional** |
| **string branch** — `StringDtype` | | | |
| S10 | `storage` equal and `(na_value is pd.NA)` parity | `STRING_POLICY` | binary |
| S11 | `_array_bytes_equal(isna(l), isna(r))` | `STRING_MASK` | binary |
| S12 | present cells: `type(a) is str and type(b) is str and a == b` | `STRING_VALUE` | unary + binary |
| **object branch** — `is_object_dtype` | | | |
| S13 | per cell `_encode_object_scalar(a) == _encode_object_scalar(b)` | `OBJECT_VALUE` | binary, plus unary encodability (`UNSUPPORTED_OBJECT`) |
| **native branch** | | | |
| S14 | `not isinstance(dtype, pd.api.extensions.ExtensionDtype)` | `UNSUPPORTED_EXTENSION_DTYPE` | unary |
| S15 | `_array_bytes_equal(to_numpy(l), to_numpy(r))` — type, dtype, `not hasobject`, shape, bytes | `NATIVE_BITS` | unary + binary |

Forty-four rows, under the **twenty-two** refusal codes
`survey_population_replay.py` raises (counted off the module, not off this
table). The admitted dtype set is narrow, which bounds the problem: `_series`
refuses `CategoricalDtype`, `Float64Dtype` and `DatetimeTZDtype` with
`UNSUPPORTED_EXTENSION_DTYPE` (measured), so only masked integer/boolean,
`StringDtype`, `object` and plain numpy dtypes get through.

The forty-four rows are a reading of the code, not a census of every mutation
that can reach it: §5's battery drives 109 comparisons, over twenty of the
twenty-two codes plus eleven pairs both paths accept, and two codes are
unreachable (`FRAME_TYPE`, because `Population` validates its own frame;
`STRING_POLICY`, because `StringDtype.__eq__` already compares storage and
`na_value`, so `SERIES_DTYPE_OR_LENGTH` fires first).

## 2. Which of them `_population_stamp` already folds

`_population_stamp` (`survey_atomic_geography.py:230-268`) folds, in order:
`canonical_json({"frame": _frame_identity(frame), "version", sorted owners,
weight_kind items, mass_ledger asdict})`; then, for every entity column in
order and then the strata, a length-prefixed `_storage_parts(series,
slice(None))` pair (physical values, null bitmap); then, per design-weight
entity, `canonical_json((entity, str(dtype), shape))` and `values.tobytes()`.

| group | folded? | note |
|---|---|---|
| P2 version, P3 owners, P5 weight_kind, P9 mass ledger | yes | in the header `canonical_json` |
| P10 design key order, P11 design bytes | yes | the design loop |
| F2 schema, F3 entities, F8 weighted entities | yes | through `_frame_identity` / `encode_us_frame_context` |
| F5 metadata, F7 mass log | yes, but through `_normative_metadata` and `_json_data`, not `_encode_frame_metadata` | a **different** normalisation from the one the comparison uses |
| F12/F13 weights | yes | `_frame_identity` folds kind, dtype, shape and `tobytes()` |
| S5/S8 masks and present values | yes | `_storage_parts` |
| S15 native bytes | yes | `_storage_parts` |
| A4 axis names | partly | `_frame_identity` folds index `names` and the strata name; not through `_axis_name_payload` |
| **S9 directional null backing** | **folded the wrong way** | it folds the masked `_data`, so it **refuses** what the comparison accepts — §0 |
| **F10 `DataFrame.flags`** | **no** | nothing folds it |
| **A1/A2 exact index class and `_comparables`** | partly | `type(index).__name__` only; not the qualified class, not `freq` |
| all unary type assertions (P1, P4, P6, P7, P8, F1, F4, F6, F9, S1, S6, S12, S14) | no | they are assertions, not content |

So the honest summary is: `_population_stamp` folds most of the *content*, gets
the one directional predicate backwards, misses `DataFrame.flags` outright, and
— unavoidably — folds none of the unary assertions.

## 3. Which it does not, and how each is preserved

Three mechanisms, applied in this order.

**(a) Every unary assertion is made on arrival, in the seal constructor, with
the same refusal code.** `replayed_population_seal(population)` runs P1, P4,
P6, P7, P8, F1, F4, F6, F9, S1, S6, S12, S14, A3 and the two encodability
assertions (`UNSUPPORTED_OBJECT`, `UNSUPPORTED_AXIS_NAME`) against the one
population it is given, raising the identical
`SURVEY_POPULATION_REPLAY_<CODE>` string. Four of them (P4, P6, and the
`actual`-side halves of the rest) are written today as assertions about
`actual` only; sealing both sides makes them fire on `expected` too. That is
strictly more closed, it fires earlier, and no code changes.

Two are re-expressed rather than dropped, because they are not properties of
one side:

* **S2/S3, the dtype pair.** Instead of `type(a.dtype) is type(b.dtype) and
  a.dtype == b.dtype`, the seal folds a **dtype token**
  `(type(dtype).__module__, type(dtype).__qualname__, str(dtype),
  getattr(dtype, "storage", None), getattr(dtype, "na_value", …) is pd.NA)`.
  Token equality must be equivalent to the pair for every admitted dtype; the
  battery proves it by sweeping a dtype census and asserting the equivalence
  both ways. `str()` alone is not enough — `StringDtype("python")` and
  `StringDtype("pyarrow")` both spell `string` — which is why `storage` is in
  the token; and `CategoricalDtype`, whose `__eq__` is not an equivalence
  relation, never reaches here because S14 refuses it.
* **A1/A2, `type(...)` and `Index.identical`.** The seal folds the **qualified
  class** (`__module__` + `__qualname__`, rather than `_frame_identity`'s bare
  `__name__`), the dtype token, and a digest over every name in
  `type(index)._comparables`. That last part closes a gap neither existing
  seal has: `pd.DatetimeIndex._comparables == ['name', 'freq']` (measured at
  pandas 3.0.3), so two `DatetimeIndex`es with identical values, dtype and
  name but different `freq` are **not** `identical()` while their bytes are
  equal. `name` is folded through `_name_bytes`; any other comparable is
  folded through `_name_bytes` when the store codec accepts it and through
  `repr()` otherwise. §6 states the one residual risk that `repr` fallback
  carries.

**(b) Every binary comparison folds exactly the bytes that comparison
compares.** Not an approximation of them, and not a different normalisation:
F5 folds `canonical_json(_encode_frame_metadata(metadata))` because that is
what `same_replayed_frame` compares — *not* `encode_us_frame_context`'s
`_normative_metadata`, which is what `_frame_identity` folds. S15 folds
`to_numpy(copy=False).tobytes()` with its dtype and shape, so NaN payload bits
and `-0.0` survive, which `_cell`'s `None` mapping destroys. F10 folds
`allows_duplicate_labels` — generically, over `type(flags)._keys` (measured:
`{'allows_duplicate_labels'}`), so a pandas release that adds a flag is folded
without an edit here.

**(c) The one directional predicate, S9, is folded as three values rather than
one digest**, which is what makes an exact reproduction possible at all. Per
series the masked branch records

* `mask_digest` — the null mask bytes (S5),
* `present_digest` — the present values' bytes with dtype and shape (S7, S8),
* `full_digest` — the whole `_data` buffer's bytes,
* `null_backing_canonical` — `bool(np.all(data[mask] == 0))` for **this** side,

and the seal comparison requires
`present_digest` equal **and**
(`actual.null_backing_canonical` **or** `full_digest` equal),
which is `_array_bytes_equal(ld, rd) or np.all(rd[rm] == 0)` term for term, per
series, with the same direction and the same `NONCANONICAL_NULL_BACKING` code.
A single population-wide flag would **not** reproduce it: today's disjunction is
evaluated per column, and one non-canonical column must not force whole-frame
byte equality on the others.

The seal record is therefore a nested structure of 32-byte digests and small
tokens whose size is **O(columns), not O(rows)** — for a US frame of six
entities and a few hundred columns, tens of kilobytes per node against about
1.5 GiB for the retained population at 1/10.

## 4. What is no longer proved

**Nothing about the content.** Every one of the forty-four discriminations in
§1 still refuses the same defect with the same code; §5's battery is the proof
obligation and it is discharged mutation by mutation, in both directions.

Four things change in character rather than in coverage, and this note states
them rather than leaving them to be found. Three were established by an
adversarial pass over the finished seal, not by writing it.

1. **Byte equality becomes sha256 equality.** Today `same_replayed_population`
   compares the bytes; the seal compares a 256-bit digest of those bytes. This
   is the same substitution the whole runtime already rests on —
   `_population_stamp`, `_frame_identity`, `preparation_sha256`, every node key
   and every store address are sha256 digests of content — so it introduces no
   assumption the build does not already make. It is not nothing, and it is
   written here so the owner can see it. Every predicate that is **not** a byte
   comparison keeps the object it applies to and the identical `is`/`==`: the
   dtype, its class, the index class, the axis name, the `WeightKind`, the
   flags. That is why `np.longlong` against `np.int64` — equal dtypes, equal
   `str()`, equal `dtype.str`, equal bytes, different dtype class — still
   refuses `SERIES_DTYPE_OR_LENGTH`, which a token built from spellings would
   have missed.
2. **Unary assertions fire earlier.** A `MultiIndex`, an unencodable object
   cell, a `Float64` column or a non-`str` owner value is refused when the
   population is observed rather than when it is compared, with the same code.
   For `expected`-side defects that is unchanged in timing; for `actual`-side
   defects the refusal moves from after `run_graph` returns to inside the
   observer callback, which `run_graph` documents as refusing the run
   (`executor.py:2453-2454`).
3. **A one-sided defect now fires before a two-sided difference.** When a
   population carries both — an unsupported dtype *and* a changed version, say
   — today's comparison reports whichever its own order reaches first, and the
   seal reports the one-sided one, because the seal is built before anything is
   compared. The same set of codes; a different precedence between them. No
   pair in the battery carries two defects at once, and none is manufactured.
4. **On an object-dtype axis, a difference reports `AXIS` rather than
   `OBJECT_VALUE`.** `_axis` refuses `AXIS` when `Index.identical` fails, and
   `identical` runs `array_equivalent` over the index's own values, which for
   an object axis is an element-wise `!=` over arbitrary Python objects: it
   holds `True` equal to `1` and `-0.0` equal to `0.0`, which the store's
   scalar codec spells apart. No digest reproduces an arbitrary `!=`, so the
   axis folds the codec's bytes, which is never weaker than `equals` and is
   stricter on exactly those pairs — they refuse under `AXIS` instead of under
   the `OBJECT_VALUE` they reach today. Every non-object axis kind is exact,
   including the three `array_equivalent` is byte-tolerant for: `float`,
   `complex` and `bool` all collapse NaN payloads, NaN sign, signed zeros and
   bool bytes outside `{0, 1}` before folding, so a byte-only difference still
   reaches `NATIVE_BITS`.

None of the four is a narrowing of what a run proves about the data — every
defect is still refused, and item 4 is a refusal moving from one code to
another — so this note does not stop and ask. The report §10's *stated
mechanism* was wrong and this note
says so in §0 and replaces it; the *decision* — compare content seals, not
objects — is implemented as chosen.

## 5. The proof obligation

For every row of §1, a test that mutates exactly that property in one of two
otherwise-equal populations and asserts:

* `same_replayed_population` refuses, with its code — this half already exists
  for most rows in `packages/microcosm-build/tests/test_us_survey_population_replay.py`
  and is extended to cover the rest;
* the **seal** comparison refuses the same mutation with the same code.

Plus, in both directions, the pairs today's comparison **accepts** — the store
round trip of §0, the object-scalar equivalences of
`test_object_scalar_equivalence_uses_actual_store_types` — which the seal must
also accept. A mutation the seal misses is a finding, not a test to delete.

## 6. Fail-closed after the change

**Every refusal code that exists today still fires on the same defect.** All
twenty-two are raised by the seal constructor (one-sided) or the seal
comparison (two-sided), with the same `SURVEY_POPULATION_REPLAY_` prefix and
the same suffix. No code is retired and none is added, and §5's receipt records
which comparison reached which.

**`FINANCIAL_NODE_POPULATION_CHANGED` after the change** means: *for a node
whose `Population` object the run still retains — the declared-consumer roster —
the object's in-process `_population_stamp` differs from the one recorded when
the run was issued; and for every other node, the seal record the run retains
is not the one recorded at issuance.* Both arms refuse the same class of defect
the one arm refuses today — something the run retained about a node's
population changed after issuance — and neither arm is vacuous, because the
seal record is re-digested rather than merely re-read.

**The declared-consumer roster.** `result.financial_population is
observed[final_node]` (`graph_atomic_survey_financial.py:1897` at the base,
`:1986` here) is an
identity check, so the objects a caller holds must be the objects the observer saw. The
base financial run declares three: the final node (`financial.ATTACH_NODE`, or
`_tax_module().GATE_NODE` when the tax rebase is enabled),
`financial.ATTACH_NODE` itself (`legacy_population`, `:1821`), and the property
population (`observed[final_node]` before the rebase moves `final_node`,
`:1818`). Those three are retained as objects and stamped exactly as today.
Every other observation is sealed on arrival and dropped.

**The completion host is a consumer of the base run's whole roster, and this
note says so because reading the code was not enough to find it — a test was.**
`graph_survey_completion_host` runs its own `run_graph` with its own observer
(`:781-794`), retains every observation, and pins their object ids in its own
identity fence (`:543-545`, `:572`). Those parts do not move. But at `:814-816`
it reads the **base** run's `state.node_populations` as its own expected
populations for the base-graph nodes, compares them against its own
observations, and hands them to `atomic._states`, which reads
`population.frame.table(entity)` for every structural node
(`graph_atomic_survey_population.py:85-96`). Seals cannot serve that.

So the base run takes a private flag, `_retain_every_node_population`, whose
only caller is `run_atomic_survey_financial`'s own recursive base call when
`child_property is not None` (`:1339-1357`). With it set, the declared-consumer
roster is every node and the run retains exactly what it retains today. The
flag and `child_property` are mutually exclusive, which the runner requires.
`_node_population_seals` accepts both shapes.

This is the one place where the memory saving does not apply, and it is the
completion path rather than the base 19-node path the measured wall belongs
to. Removing it means giving the completion host the same seal treatment and
finding a frame-free form for `_states`; that is a second change with its own
argument.

**The residual risk this note will not hide.** The `_comparables` fallback in
§3(a) uses `repr()` for a comparable the store's axis-name codec will not
encode. `repr` is not guaranteed injective with respect to `==`, so a pair of
`==`-equal, differently-`repr`'d comparables would make the seal refuse where
today's `identical()` accepts. No index class in this runtime reaches it: US
frames carry `Index` and `RangeIndex`, whose `_comparables` is `['name']`
(measured), and `graph_context.encode_us_frame_context` requires group entities
to carry a default unnamed `RangeIndex` (`:130-143`). The seal folds the
comparables tuple itself, so a future index class with a wider one cannot pass
unnoticed.

## 7. The `microcosm-graph` half

`_observer_snapshot` (`executor.py:356-408`) detaches by round-tripping every
entity and link table plus the strata through `pickle.loads(pickle.dumps(...,
protocol=5))` — one full independent copy per reached node, cache hits included
(`:2773-2774`). An observer that only seals never retains and never mutates, so
it needs no detached copy.

`run_graph` gains `_population_observer_detach: bool = True`, private and
paired with `_population_observer`. When it is `False` the observer receives the
live admitted population and no snapshot is allocated. Default behaviour is
unchanged, the keyword enters no key, no receipt and no cache record, and the
change imports nothing from the US runtime.

**It is deliberately not called `population_retention`.** That keyword existed
on PR #893's original branch (commit `6c0f24c77`, layer 08) and was dropped; it
made the **manifest's attached population views** lazy against the content
store, which is a different target from the observer's per-node detached
snapshots (`docs/us-native-scale-transport.md` §4). Reusing the name for this
would invite exactly the confusion that document warns about.

Invariants that do not move, each with the test that pins it
(`packages/microcosm-graph/tests/test_graph_executor.py`):

| invariant | test |
|---|---|
| exactly once per reached node, in `compiled.order`, cold and on hits | `:4130-4166` |
| an observer's exception refuses the run with the observer's type | `:4172-4179` |
| a mutating, retaining observer cannot move a key, a manifest or a store object — **in the default detaching mode** | `:4183-4248` |
| with no observer, no snapshot is allocated | `:4368-4374` |

The new tests are the shape of the last: with an observer **and**
`_population_observer_detach=False`, a monkeypatched `_observer_snapshot` that
raises must never be called; the run's manifest key and every store object must
equal a no-observer run's; and the observer must receive the same object the
executor holds, which is the mode's whole point and is what the caller is
declaring it will not mutate.

That declaration is the mode's one honest cost: in this mode the executor no
longer *enforces* that the observer cannot reach execution state, it *trusts*
the caller's declaration. The default mode still enforces it, and this mode is
opt-in and private.
