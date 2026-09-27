# Reuse ACS serial membership lookups

This native-lane continuation starts at `9af56aa8a`. It prepares each exact
string-key lookup once per person-archive read, then checks every incoming
chunk against that lookup. The pandas Arrow path previously rebuilt its
complete comparison set for each chunk. The completed native attribution
sample assigned about 108 cold / 119 replay CPU seconds to that operation;
these are statistical estimates, not function timers or promised savings.

An object-dtype pandas Index retains its hash table for the duration of the
archive call. Missing chunk values remain unmatched because the optimized
keys are exact strings. Any mixed key set or other Series dtype uses the
original pandas `isin` path. Age validation, complete orphan checking,
household roster validation and selection order are unchanged. No source row,
archive parse, validation result or issued authority is cached. The native
coverage owner explicitly pins the reviewed reader hash; altered source bytes
still refuse with `UNREVIEWED_PREPARATION`.

## Evidence

- `tests-with-owner.txt`: 185 passes, covering the new differential tests,
  existing ACS PUMS behavior, coverage authentication and source compilation.
- Differential cases include both pandas string storage backends, both missing
  policies, mixed/non-string fallback keys, duplicate indices, names, Unicode,
  literal leading-zero IDs, chunk changes and exact orphan/age error messages.
- Actual native `_producer()` admits the reviewed implementation; a separate
  changed-source-byte test confirms the existing refusal remains active.
- `ci-inventory.txt`: the new flat test file is included in the CI inventory.
- Ruff check/format and `git diff --check` pass.

`benchmark.py` uses invented arrays only: 200,000 comparison keys and twenty
20,000-row chunks on pandas 3.0.3 / PyArrow 25.0.0. Separate processes produced
`original.json` and `prepared.json`, and every result matched literal pandas
`isin`. Including preparation, timed CPU was 6.691459 seconds original and
0.052461 seconds prepared. The prepared first lookup took 0.012154 CPU seconds
and reused the same index engine for the remaining nineteen calls.

This is a focused microbenchmark, not a native build measurement. The prepared
Index's deep estimate is 17,627,112 bytes including Python strings shared with
the existing key set; it is not incremental RSS. The process RSS observations
include imports and the untimed pandas oracle. The optimization adds a local
O(key-count) index and O(rows-per-chunk) temporary lookup buffers, and does not
establish lower memory use or full-source capacity. Key preparation also occurs
before archive/member validation, so malformed archives can incur that bounded
extra allocation before the same validation refusal.

## Reproduce

The recorded environment is the existing native-lane Python 3.14.4 virtual
environment, reused without installs or lock changes. All six Microcosm source
shards are explicitly loaded from this checkout; `benchmark.py` checks its
reader import origin. Run it with isolated Python (`-I -B`), then choose
`original` or `prepared`. No real archive or population data is read.

The first test-launch attempt lacked `-I` and encountered an unrelated
`/private/tmp/dis.py` shadowing the standard library before test collection.
Isolated Python corrected the harness. The subsequent expected red test failed
because the lookup helpers did not yet exist; the final suite above passes.

This branch changes producer-bound provenance and may change graph keys.
Compare all/compact retention at the same combined source revision, and record
cross-revision semantic equivalence separately. No actual native replay,
certification, default change, merge or publication is claimed by this change.
