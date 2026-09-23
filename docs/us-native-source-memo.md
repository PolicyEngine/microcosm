# Persistent memo for US source derivations

A native US run spends most of its CPU outside the graph, re-deriving the same
values from the same pinned source archives. This note records where that cost
is, what `microcosm.build.us_runtime.source_memo` memoizes, the rules that keep
a recalled value exactly equal to a recomputed one, and what was measured.

## Where the outside-graph CPU goes

`experiments/native-source-auth-memo/attribute_outside_graph.py` partitions
every sampled stack in the `outside_graph_executor_calls` bucket of the
2026-09-19 1/1000 attribution run at `9af56aa8a`. The input is the harness's
`completed-aggregate-report.json`, SHA-256 `4e7be8ad…`. The script's output,
`attribution-9af56aa8a-1-1000.json`, reproduces byte for byte from that report.
The values are statistical process-CPU estimates at a 0.25 s sampling
interval, not exact function CPU.

| Phase | Process CPU-s | Outside graph | Share |
|---|---:|---:|---:|
| Cold | 1,999.4 | 1,681.6 | 84.1% |
| Required replay | 1,652.1 | 1,624.7 | 98.3% |

Each sampled stack is charged to exactly one issuer: the first owner entry
point found on it, taken in a fixed precedence order. Cold phase:

| Issuer | CPU-s |
|---|---:|
| ACS native coverage (`issue_acs_native_coverage`) | 469.4 |
| ACS source catalogue (`issue_acs_source_catalogue`) | 467.5 |
| ASEC source catalogue | 176.8 |
| ASEC native population | 174.6 |
| Atomic geography requalification | 108.4 |
| Predictor requalification | 100.8 |
| Preparation assembly and validation | 68.7 |
| Verification-epoch close | 43.4 |
| Other (selection, runner, borrow checks, seals) | 72.1 |

The two ACS issuers account for 936.9 of the 1,681.6 CPU-s. Their cost is
dominated by work that is a pure function of the archive bytes:

| Function (inclusive, cold; overlapping) | CPU-s | Call sites |
|---|---:|---:|
| `acs_housing_universe_source._reconstruct` | 381.5 | 2 |
| `acs_housing_universe_source._archive` | 220.1 | 2 |
| `acs_population_catalogue._collect` | 216.5 | 1 |
| `acs_person_coverage_authentication._inventory` | 153.9 | 3 |
| `acs_pums.load_acs_pums_tables` | 148.9 | 1 |
| `acs_housing_universe_source._select` | 121.0 | 2 |

The ASEC figures predate the CSV guard fast path (`b4f9b8eeb`), which this
branch's base `47960af43` carries. In the `9af56aa8a` profile,
`asec_coverage_authentication._capture` took 221 inclusive CPU-s across its two
call sites. Re-timed on the same staged inputs at this branch, each of the three
ASEC person-member captures (about 280 MB each) took 1.65 to 1.68 CPU-s. The
ACS work is what remains.

## What is memoized

Four derivations, each keyed by the exact bytes it reads:

| Namespace | Stands in for | Key inputs |
|---|---|---|
| `acs_housing_universe_source.full_projection` | `_archive` of both archives: the canonical full lexical projection and member inventories | both archive SHA-256s |
| `acs_housing_universe_source.selection` | `_select` and the receipt built from it | full projection and inventory digests, pins, the exact selection tuple's digest, implementation and definition digests |
| `acs_person_coverage_authentication._inventory` | one archive's member inventory and selected rows | archive SHA-256, role, digest of the sorted selection |
| `acs_population_catalogue._collect` | the catalogue records, vacancies and counts | full projection digest, both archive SHA-256s |

In one cold 1/1000 preparation, the housing owner runs twice: once for the
catalogue and once for the selected native population. The second
`_archive` pair and the repeated selected inventories are recalled within the
run. A later run recalls all four derivations.

Owners change nothing else. They still capture and pin-check their sources,
write and re-read the projection file, construct and seal every issued
object, and run every live-object, mutation and producer check. The recorded
producer and implementation identities change only because the owner files
changed.

## Rules

**Keys.** A key is `sha256("microcosm.us.source-memo.v1\0" + canonical(document))`.
The document holds:

- the namespace;
- a runtime identity: interpreter version, optimisation level and platform,
  the zlib runtime, digests of the stdlib `csv`, `json` and `zipfile` modules,
  the binary that provides `_csv`, `_json` and `zlib`, and the memo module
  itself;
- the owner's code identity;
- every input as `{role, sha256, bytes}`;
- the canonical parameters.

`canonical` is sorted, compact JSON encoded as strict UTF-8. ASCII escaping
would spell a lone-surrogate pair and the astral character it resembles
identically. Strict UTF-8 refuses to encode the pair, so the value never
becomes a key. File inputs are hashed through one no-follow descriptor at call
time. Stat metadata is never trusted.

**Code identity.** Each owner passes the identity its receipts already record:
the scoped implementation hash for the housing owner, and `_producer()` for
coverage and the catalogue. It adds `live_code(*modules)` over the modules the
derivation executes, which records:

- each module file's SHA-256;
- a check that every function and method the module defines still runs the
  code compiled from those bytes;
- the identity of every function bound in the module's namespace;
- the module's immutable constants, such as limits, pins and field lists.

A monkeypatched constant or function changes the key. Loaded code that no
longer matches its file raises, and the memo is bypassed.

**Entries fail closed.** An index file `index/<kk>/<key>.json` names
content-addressed `blobs/<dd>/<sha256>` files. It carries an HMAC-SHA256 made
with a per-user key that must live outside the memo root, so a copied or
foreign memo confers nothing. A lookup accepts an entry only if all of these
hold:

- the index is a bounded, regular, non-symlink file in canonical form;
- its MAC verifies;
- its key document equals the requested one byte for byte;
- every blob is a regular, non-symlink file of the recorded size and SHA-256.

Anything else is a counted miss, never partial trust. The owner recomputes from
source, and the fresh entry atomically replaces the rejected one.

**Only completed, exactly representable values are recorded.** An exception
from the computation propagates unchanged and records nothing, so every refusal
keeps its code. After a miss, the code identity and file inputs are
recomputed. A value is stored only if neither moved and the owner's proof shows
that the encoding decodes to exactly the computed value. The proof checks exact
types: tuples, lists, str, int and dict order. A value outside those types, or
one with a lone surrogate, is not stored.

**Trust boundary.** Whoever holds the HMAC key is trusted to have run these
owners. A MAC-valid entry whose blobs decode wrongly is caught only if decoding
fails. Keep the key private (it is created `0600`) and outside shared storage.

## Enabling it

The memo is off by default, and an ordinary call runs exactly the unmemoized
path. There are two ways to enable it:

```python
from microcosm.build.us_runtime.source_memo import source_memo

with source_memo("/path/to/memo", key_path="/path/outside/memo.key"):
    ...  # native preparation or graph run
```

or set these environment variables:

```bash
export MICROCOSM_US_SOURCE_MEMO=/path/to/memo
export MICROCOSM_US_SOURCE_MEMO_KEY=/path/outside/memo.key  # optional
```

The default key is `$XDG_CONFIG_HOME/microcosm/us-source-memo.key`, or
`~/.config/microcosm/us-source-memo.key`. It is created on first use.
`source_memo(None)` disables the memo inside an enabled context. Context
activation is a context variable, so a new thread sees only the environment
variables. `source_memo.statistics()` reports hits, misses, rejections,
bypasses, bytes stored and seconds per namespace.

The memo is a cache. Deleting its root is always safe. It never prunes itself.
A code change produces new keys, and old entries stay until the root is
removed.

## Evidence

`packages/microcosm-build/tests/test_us_source_memo.py` covers:

- **Memo behaviour:** activation and isolation, content-addressed hits, and
  key sensitivity to bytes, code and parameters.
- **What is never recorded:** refusals, values that cannot be represented
  exactly, and inputs or code that moved during the computation.
- **Bypass:** identities that cannot be formed skip the memo.
- **Tampering:** blob digest, size or absence; index MAC, fields, form or
  garbage; symlinks; a foreign key; a misfiled entry; a decode failure. Each is
  a counted miss that the next store repairs.
- **`live_code` binding.**
- **Owner identity:**
  - The housing source, the ACS catalogue and the whole ACS+ASEC preparation
    payload are byte-identical with the memo off, cold and warm.
  - A warm preparation executes none of `_archive`, `_select`,
    `_inventory_uncached` or `_collect`.
  - A monkeypatched limit, or changed archive bytes, is never answered from
    the memo.

`experiments/native-source-auth-memo/bench_acs_owners.py` times the ACS owners
on the actual staged archives, in one process per phase. The receipts,
`acs-owner-bench-{off,cold,warm}.json`, are summarised under "Measured" below.

## Measured

See `experiments/native-source-auth-memo/README.md` for the owner-level
benchmark on the actual archives, the machine conditions it ran under and the
status of the 1/1000 graph-run measurement.

## Not memoized

- `acs_pums.load_acs_pums_tables` (148.9 inclusive CPU-s in the profile) and
  `read_acs_person_coverage_columns`. They return pandas frames. An exact frame
  codec and proof would need dtype-exact round trips, including string storage.
- The ASEC owners. Their capture cost was already cut at the base.
- Every live-frame seal and requalification. These digest live objects, not
  source bytes, so there is nothing persistent to key on.
