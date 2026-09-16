# Build lane: verify native sources once per run

Branch `native-verify-once`, worktree `~/PolicyEngine/_worktrees/microcosm-verify-once`,
base `f7bb88525a78786f91bc3ebe2083ef4b1c85de18` (PR #893 head — `git rev-parse HEAD`
and `git log -1` checked before any edit). 2026-09-15/16.

**Where this report is.** The repository's root `out.md` is a tracked file
holding the Amendment 19 lane's committed report, and root `PROGRESS.md` is
cumulative. Writing this lane's report to either would delete another lane's
committed content, so the report is this file and the lane's journal is an
appended section of `PROGRESS.md`.

**What this is not.** Nothing here is a statement about dataset quality,
calibration or release eligibility. Every measurement is descriptive; none is a
build, a certification or a release artifact.

## The problem

`~/PolicyEngine/_recovered/pilot-runs/native45-v5/out.md` §2, measured by stack
sampling on real runs: on the 19-node financial graph at `Fraction(1, 1000)`
(3,168 expanded households, 5,362 s wall), node execution is 1,143 s and the
remaining ~79 % is admission and verification of 3.48 GiB of staged source,
repeated. QRF fits inside that run cost 0.06–0.07 s each.

The design note is [`docs/us-native-verification-once.md`](../../docs/us-native-verification-once.md):
per mechanism, the guarantee today, the change, why the guarantee survives and
the new cost class. It was written and committed before any capsule changed
(`d9136a20e`).

## Per mechanism

### 1. `AuthenticatedSurveyPopulationPreparation._checked()`

Every borrow re-ran the whole authentication — both source catalogues, the ACS
native coverage binding, the nested ASEC native population, the ten-file roster
re-hash and every pure seal — and the graph registers
`verify_survey_population_preparation` as a **per-node population observer**, so
a run paid one of these per executed node on top of every kernel borrow.

**Guarantee, in two sentences.** Inside a `verification_epoch()` every borrow
still pays, in full and unmemoised, the live authority, the attached owner
payloads, `_encode(_producer())` and `_file_stats` over the whole source roster,
so every producer change and every on-disk change that moves any of five stat
fields still refuses at the same borrow it refuses at today. What a signature
cannot see — an in-place write into a live buffer, or a value written into a
frozen plan row — is closed by an unconditional full re-validation when the
epoch closes, before the run returns anything; outside an epoch nothing is
memoised at all.

**Mutation tests.**
`test_a_source_appended_to_mid_epoch_refuses_at_the_next_borrow`
(3 parametrisations), `test_a_source_truncated_mid_epoch_refuses_at_the_next_borrow`
(2), `test_a_source_rewritten_in_place_mid_epoch_refuses_at_the_next_borrow`,
`test_a_file_added_to_a_source_directory_mid_epoch_refuses`,
`test_a_touched_source_refuses_on_its_stat_identity_alone`,
`test_a_change_a_signature_cannot_see_refuses_when_the_epoch_closes`.
Each asserts the refusal **twice**: at the borrow that follows the mutation, and
again when the epoch declines to close over it.

**Before/after CPU.** Unit level, from
`test_an_epoch_validates_once_and_reuses_it`: six borrows of one preparation
cost six complete validations before (one `_source_files` pass each — the whole
3.48 GiB roster) and one after, plus one unconditional pass when the epoch
closes. Five of six borrows are memo hits; `_producer()` still runs on all six.

### 2. `AuthenticatedAsec2024NativePopulation.frame`

Every `.frame`, `.context`, `.receipt` and `.to_bytes()` was a full
re-authentication including a SHA-256 of all seven ASEC source files.

**Guarantee, in two sentences.** `_encode(_implementation())` still runs on
every borrow, and the signature carries the stat identity of all seven source
files *and of their parent directories*, which `_file_identity` does not cover
because it opens `O_NOFOLLOW` on the final component only — so a roster change
reaches a memo that never lists the roster. The capsule's epoch closes **before**
its owner's, so the owner's final validation reaches the complete file check
rather than the memo.

**Mutation test.**
`test_the_native_capsule_still_refuses_a_changed_source_inside_an_epoch`
(`SOURCE_FILE_CHANGED` at the borrow, and again at the close). The existing
`test_changed_original_source_refuses_existing_borrow` is unchanged and still
passes: it runs outside an epoch.

**Before/after CPU.** `test_the_native_capsule_validates_once_per_epoch`: three
`validate()` calls cost three complete seven-file passes before and one after,
plus one at the close.

### 3. Per-node source content keys in the executor

`executor.py:2452-2457` re-derived `source_content_key` for every source a
cold-executed node declared; `keys._directory_identity` `read_bytes()`es every
file of the tree.

**Guarantee, in two sentences.** A cached key is reused only while the path's
stat signature — for a directory, its own identity plus the relative name, type
and identity of every entry `_directory_identity` would walk — is identical to
the signature taken both immediately before and immediately after the read that
produced the key, so every mutation that moves any stat field still refuses at
the same node, before that node's `_write_node`. Every source is then re-derived
in full, cache bypassed, before the manifest is built, which also catches a
change made during a node that declares no source — something the per-node check
has never seen.

**Regression tests.** The refusal had **no test anywhere in the repository**
before this branch. `test_graph_executor_source_identity` now pins it:
`test_a_source_rewritten_by_its_own_node_refuses_that_node` (and asserts the
store gained no object), `test_a_file_added_to_a_directory_source_refuses`,
`test_a_file_removed_from_a_directory_source_refuses`,
`test_a_rewritten_source_still_refuses_when_only_its_bytes_moved`,
`test_a_source_changed_during_a_source_free_node_refuses_at_run_end`.

**Before/after CPU.** `test_each_source_is_read_twice_per_run_not_once_per_node`
and `test_an_unchanged_source_is_never_re_read_by_a_node`: two full derivations
per run regardless of how many nodes declare the source. The
`packages/microcosm-graph` suite fell from **108.28 s to 71.81 s** on the same
machine — a synthetic toy source, so the real saving scales with source bytes.

**The record.** `RunManifest.source_identities`, attached exactly like
`populations` and `mass_ledgers`: `repr=False`, `compare=False`, outside
`content_addressed`, outside `to_json`, outside every node receipt and cache
record. `test_the_manifest_records_the_run_end_identities_without_moving_anything`
asserts the manifest's key, its JSON and its content-addressed body are
identical with and without it.

### 4. The ACS record fence

`acs_person_coverage_authentication._records` was a `for byte in block` loop
over 4,096-byte blocks, fencing 2.4 GB of ACS person CSV.

**Guarantee, in two sentences.** The state machine is unchanged — quote parity
still persists across records, CR/LF/CRLF are still preserved, and both ceilings
are still charged in the same order — only the boundaries are now located with
`bytes.find` and `bytes.count`. A record no longer than the smaller of the two
live ceilings cannot have violated either, because the token counter resets at
every record start; anything longer is replayed byte by byte, so the refusal
code and the byte it fires on stay the fence's own.

**Equality proof.** `test_us_acs_record_fence_scan` compares the shipped fence
against a verbatim copy of the byte loop on every input, comparing the yielded
record sequence and the refusal together: 20 boundary cases, 6 block-boundary
cases, 8 ceiling cases on both sides of every cap, 4,200 random strings, 300
row-shaped inputs, and **all 9,331 strings up to five bytes over the branching
alphabet under each of five ceiling settings — 46,655 exhaustive comparisons**.
On the real staged archive
(`experiments/native-verify-once/record-fence-parity.json`, produced by
`record_fence_real_archive.py`):

| member | bytes | records | digest | byte loop | chunked scan | speedup |
|---|---|---|---|---|---|---|
| `psam_pusa.csv` | 1,226,543,489 | 1,743,752 | `da955ca2…347fc827` identical | 116.72 s | 2.38 s | 49.1× |
| `psam_pusb.csv` | 1,178,924,624 | 1,679,138 | `9bbf6af8…90184bba` identical | 110.14 s | 2.25 s | 48.9× |

**226.86 s → 4.63 s per pass**, 3,422,890 records, digests identical.

### 5. The kernel context digest

`_update_series` boxed every value of a plain float, integer or boolean column
into a Python object and framed them one at a time; on pandas 3.0.3 such a
column is a `NumpyExtensionArray` with no `_data`/`_mask`, so the existing fast
path never applied to it.

**Guarantee, in two sentences.** `_object_stream` emits the identical framed
bytes — the same `object` dtype header, the same shape header, the same
per-value length prefixes and payloads — and returns `None` for every column
kind it cannot reproduce exactly. Both `_context_digest` calls per node remain,
because the second one is the "did the kernel mutate its context" guard and
reusing its result would delete the guard.

**Byte-equality proof.** `test_graph_executor_series_stream` compares the live
helper against a verbatim copy of the pre-change body **at the byte level**, not
only at the digest: 19 column kinds, the float specials (signed zero, both
infinities, the subnormal extremes), non-canonical NaN payloads with their sign
bit, both `int64` endpoints, empty/sliced/strided columns, labelled indexes, and
a 200-frame random sweep over ten kinds.

The first version of the fast path mistook a `Categorical` for an integer column
because `Categorical._ndarray` is its codes array; that is now a named test,
`test_categorical_is_not_mistaken_for_its_integer_codes`, and the shipped guard
keys on `series.dtype` being a plain numpy dtype rather than on any private
attribute.

**Before/after CPU**, 6,928 × 240 frame: **0.466 s → 0.094 s**. Per value:
float64 182.6 → 17.9 ns, bool 225.6 → 13.9 ns, int64 383.3 → 119 ns.

## Pins re-derived

| pin | old | new | command |
|---|---|---|---|
| `acs_native_coverage_binding._ACCEPTED["acs_person_coverage_authentication.py"]` | `475aa795c8a5b49a0dd3405a0877866dddd03012a1f2b9743447fab1fe85bcff` | `9ec68721a4cf480ef412c51ab354db9000eb7a7989e35e6e09b1574d88e8e49f` | `shasum -a 256 packages/microcosm-build/src/microcosm/build/us_runtime/acs_person_coverage_authentication.py` (re-derived after the final formatting pass; a stale pin refuses with `UNREVIEWED_PREPARATION`) |
| `graph_implementation_inventory.json` → `survey_population_preparation.py` → `unbound_uses_sha256` | `29c09f6fcb25ce8bc6111dd0dc76f3501d9a6b635929e94b80835889d296ef91` | `d114117ca19e9b866982147e5497d601ed91ca603d0568730d4051bdd4005910` | `graph_implementation._dependency_contract(payload, name, _covered_imports(name, inventory))` — the module's own generator |
| `graph_implementation_inventory.json` → `asec_2024_native_population.py` → `unbound_uses_sha256` | `71463df4f645cd2a98f537a212187b4d1e52bffcc4d2317defe6968d7f287608` | `7bb20439da26f75b48551e21e304c4bb7f3983ee07ef78ed804411640d58cdf4` | same |

No other pin moves. Verified by recomputing **every** contract in the inventory
through `graph_implementation._dependency_contract` and building all ten stage
manifests: `moved: 0`, `all stage manifests build`. In particular the three
foreign owners that gained a `_verified_source_stats()` helper
(`acs_population_catalogue`, `acs_native_coverage_binding`,
`asec_population_catalogue`) and both graph-shard modules keep their contracts
unchanged, because the additions introduce no `_RESOURCE_CALLS` name and no
non-stdlib import.

**One thing that does move, by design and not by this lane's choice.** Each US
stage's `implementation_hash` is `_digest(module bytes)` over its whole module
roster, so editing any inventoried module moves it, and with it every node key.
That is true of any change to these files, including a comment. It is not a
digest this lane computes; the digests this lane computes — frame identities,
source file digests, context digests, source content keys, seals, receipts — all
keep their exact values, and the tests above prove it byte for byte where bytes
are rebuilt.

## Test summaries, verbatim

```
$ .venv/bin/python -I -B -m pytest packages/microcosm-graph/tests -p no:randomly
766 passed, 1 skipped, 1 warning in 71.81s (0:01:11)
```

```
$ .venv/bin/python -I -B -m pytest packages/microcosm-graph/tests/test_graph_executor_series_stream.py
34 passed in 0.75s
```

```
$ .venv/bin/python -I -B -m pytest packages/microcosm-graph/tests/test_graph_executor_source_identity.py -p no:randomly
15 passed in 0.56s
```

```
$ .venv/bin/python -I -B -m pytest packages/microcosm-build/tests/test_us_acs_record_fence_scan.py -p no:randomly
52 passed in 1.08s
```

```
$ .venv/bin/python -I -B -m pytest \
    packages/microcosm-build/tests/test_us_acs_person_coverage_authentication.py \
    packages/microcosm-build/tests/test_us_acs_source_compile_cache.py \
    packages/microcosm-build/tests/test_us_acs_person_coverage_columns.py \
    packages/microcosm-build/tests/test_us_acs_housing_source.py \
    packages/microcosm-build/tests/test_us_acs_record_fence_scan.py -p no:randomly
544 passed, 5 warnings in 14.33s
```

```
$ .venv/bin/python -I -B -m pytest \
    packages/microcosm-build/tests/test_us_survey_population_preparation.py \
    packages/microcosm-build/tests/test_us_asec_2024_native_population.py \
    packages/microcosm-build/tests/test_us_acs_source_compile_cache.py -p no:randomly
112 passed, 2 warnings in 140.94s (0:02:20)
```

```
$ .venv/bin/python -I -B -m pytest packages/microcosm-build/tests/test_us_native_verify_once_epoch.py -p no:randomly
17 passed in 161.69s (0:02:41)
```

```
$ .venv/bin/python -I -B -m ruff check .
All checks passed!
```

```
$ python3 -I -B -S tools/ci_test_groups.py --verify
verification=ok
```

```
$ uv lock --check
Resolved 125 packages in 6ms
```

All four new test files land in the groups they should and none under
`[defaulted]`: `test_us_acs_record_fence_scan.py` and both graph files in
`rest` / `us-am` / `wheels`, `test_us_native_verify_once_epoch.py` in `rest` /
`us-not` / `wheels`.

*(Remaining runs and the probe measurement follow.)*
