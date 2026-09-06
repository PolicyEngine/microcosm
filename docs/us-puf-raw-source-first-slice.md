# The first PUF raw source slice

`microcosm.build.us_runtime.puf_raw_source` types one narrow, source-only
artifact directly from the pinned 2015 PUF delivery bytes. It is the first
Microcosm node that reads a raw PUF file at all, and it is deliberately the
smallest useful thing that can be read from one.

## What it produces

`microcosm.us.puf_2015_raw_return_status` v1: one opaque, self-describing
payload at **source-return grain**, 207,696 rows in delivered main-file order.

| Group | Fields | Typed as | Lexical bytes kept |
| --- | --- | --- | --- |
| identity | `RECID` | `int64`, lexically and numerically unique | 64-byte ASCII |
| period | `FLPDYR`, `FLPDMO` | `int16` / `int8` raw codes | 8 / 4 bytes |
| status | `MARS`, `DSI` | `int8`, closed observed domains | 4 bytes each |
| status | `disclosure_aggregate` | `bool`; must agree with `MARS` 0 | derived |
| status | `demographic_status` | `int8`: 1 present, 0 not provided | derived |
| weight | `S006` | `int64` **hundredths**, never divided | 32 bytes |
| demographics | `AGEDP1..3`, `AGERANGE`, `EARNSPLIT`, `GENDER` | `int8` codes, `-1` only where status is 0 | 4 bytes each |

The other **216** main columns are validated for header name, order and record
width and then dropped. No money field is parsed. The node owns no population
columns, creates no PUF person or household, and establishes no
return-to-household mapping; its host population is whatever the caller names.

## What it deliberately does not resolve

- **Period.** The header audit observed `FLPDYR` in {2012, 2013, 2014, 2015}
  while the booklet prints 2011–2014. The artifact records the raw code and the
  conflict verbatim; `period_semantics` stays `unresolved_raw_FLPDYR`.
- **Weight units.** `S006` is exact integer hundredths with
  `implied_decimals = 2` recorded. No `/100`, no float64, anywhere.
- **Aggregates.** The four publisher aggregate records stay in the artifact,
  tagged `disclosure_aggregate`, never disaggregated. A demographic record
  keyed to an aggregate `RECID` is tolerated, typed under the same domains and
  tagged; it never implies a person or a household.
- **Absence.** The 88,021 returns with no demographic record carry
  `demographic_status = 0`, the `-1` sentinel and empty lexical columns.
  Absence is never spelled as a zero code, and a documented zero inside a
  supplied row stays a real code.

## Three separations

**Structure is not provenance.** `decode_puf_raw_source` validates CSV
structure against the definition it is handed and authorizes nothing. The
production wrapper — `puf_raw_source_codecs` plus `USPufRawSourceKernel` —
binds the closed packaged pins and refuses any definition whose canonical bytes
are not the packaged file's. Caller-supplied metadata that merely looks right
authorizes no alternate source.

**Fixtures take an explicit non-production route.** `fixture_definition` builds
a definition marked `route="test_fixture"`, which must carry the
invented-fixture authority and may not restate a packaged source pin. There is
no path anywhere where an omitted or optional expected hash admits a file.

**The private cap is private.** `PUF_RAW_SOURCE_MAX_BYTES` is 128 MiB, because
the pinned main delivery is 126,034,649 bytes. `RAW_BYTES_MAX_BYTES` in
`microcosm.graph.codecs` stays at 64 MiB and `load_raw_bytes` still refuses
above it.

## The codec

`load_pinned_puf_bytes` opens with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK`, checks
the descriptor's own regular-file mode and exact size, compares path and
descriptor identity before and after the read, streams both a SHA-256 and a git
blob SHA-1 (`"blob <len>\0"` preimage), and refuses unless both match the pin.
It never consults a capture receipt or a run summary, so a forged receipt
cannot admit changed bytes.

`O_NOFOLLOW` protects **the path the codec receives**. The graph executor
resolves every source path strictly before codec lookup, so under the executor
a user's symlink is followed upstream and the codec sees the resolved file; on
a direct call a symlink is refused here. Both behaviors are tested.

## The envelope

Magic line, a 4-byte **big-endian** header length (matching the ACS rent
placement precedent), a canonical JSON header bounded at 64 KiB, then explicit
**little-endian** integer bodies and fixed-width NUL-padded ASCII lexical
bodies in declared order. Every body carries its own SHA-256 in the header and
the decoder verifies each one. Lexical allocation widths are declared in the
definition and stored in the header: `RECID` 64, `S006` 32, `FLPDYR` 8, and 4
bytes for `FLPDMO`, `MARS`, `DSI` and each of the six demographic fields. These
are explicit allocation bounds; they are not a claim that the header audit
validated a token grammar.

The full slice is 169 bytes per row, so 207,696 rows is 35,100,624 bytes. The
encoder derives that bound from the row count and refuses before allocating if
it exceeds `BODY_MAX_BYTES` (64 MiB).

## Registration

Compose the registries at the caller:

```python
codecs = SourceCodecRegistry()
codecs.register("csv-tables", load_csv_tables)
register_us_puf_raw_source_codecs(codecs)
store = ContentStore(root, codecs=codecs)

kernels = KernelRegistry()
register_us_puf_raw_source_kernels(kernels)

graph = Graph(
    country="us",
    sources=(host_source, *us_puf_raw_source_refs()),
    nodes=(host_node, us_puf_raw_source_node(population=host_node.id)),
)
```

This module is deliberately **not** imported into the generic
`us_runtime/graph_sources.py`: that import would need a real dependency
inventory update, and the slice does not need one.

## The artifact envelope is closed, and re-proved on readback

`decode_return_status` never trusts the header it is handed. The header is a
closed schema: the header, every column entry and every source pin must carry
exactly their declared key sets, the column kind is one of the two known
values, each typed column declares an **exact** dtype string — which closes
width, signedness and endianness in one comparison, so `<f8` can never stand
in for `<i8` — and each lexical column declares an exact allocation width and
body size, under a total body bound. A malformed, deeply nested, `null` or
non-ASCII header is a typed refusal, not a stray `UnicodeDecodeError` or
`RecursionError` escaping the JSON reader.

Every reported fact is then **recomputed** from the arrays that were actually
read back, through `_facts_from_arrays` — the same derivation the producer
uses — and must agree exactly. A payload cannot report 987,654,321 rows over
four real ones, or a match count, an aggregate count or a code frequency its
arrays do not carry. The readback also re-proves the slice's own invariants:
lexical and numeric spellings agree, `S006` stays a nonnegative exact integer,
absence stays the `-1` sentinel with an empty spelling rather than a zero
code, the `disclosure_aggregate` flag follows from `RECID` and agrees with
`MARS` 0, and both join keys stay unique.

This is still structure, not source authority. The decoder proves a payload is
internally consistent and, where a definition is available, that it matches
that definition's widths, domains and source pins. It authorizes nothing about
where the delivery bytes came from — the codec does that, by content — and no
signature or capability is invented. A payload declaring the **packaged**
route is checked against this module's own packaged definition, because that
one is closed and importable; a fixture-route payload has no reconstructible
definition and stays structural unless a caller passes one in, so the
production and fixture boundaries stay exactly where they were.

## The CSV acceptance profile is part of the producer's identity

`csv.field_size_limit` is process-global state that decides which deliveries
parse: lowered, a file that was admitted is refused; raised, a file that was
refused is admitted. `csv_acceptance_profile()` reads it without changing it —
nothing in this module ever sets it — exactly as `asec_income_observations`
and `acs_housing_universe_source` already do, and `USPufRawSourceKernel`
folds it into its implementation hash and records it in its receipt.

A changed limit therefore moves the node key. `resume="require"` misses rather
than hitting, `resume="auto"` recomputes and then refuses, and a restored
limit restores the original key and every hit. No run can reuse an artifact
admitted under an acceptance that no longer holds.

The packaged definition's `csv_profile.field_size_limit` used to be a
decorative fact — the decoder relied on the process default happening to match
it. `decode_puf_raw_source` now refuses with `CSV_FIELD_SIZE_LIMIT` when the
effective limit is not the pinned one, so that fact is true of the parse that
actually ran. The delimiter, quote character and doublequote flag still come
from the closed definition, and `QUOTE_MINIMAL` and `strict` are still fixed
by the reader itself.

Readback also checks the declared record expectations whenever it resolves a
packaged definition. A definition change invalidates older packaged payloads;
the supported path is to recompute under the new definition. Readback does not
silently interpret an old payload under new source semantics.

## First genuine run

Everything above is verified on invented fixtures written under `tmp_path`. No
genuine PUF, capture, receipt or microdata has been read by this slice. A first
genuine run belongs in a separate reviewed private runner, after root and peer
code review, and must:

- name its host population explicitly (a one-household invented engineering
  context is the smallest honest option; the production spine is the eventual
  one);
- record actual peak RSS in the lane receipt. The plan's 350 MB figure is an
  unmeasured estimate, not a certified bound; use a conservative operational
  ceiling and measure;
- surface an `S006` grammar refusal as a refusal receipt, not a partial
  artifact. The header audit did not establish the complete `S006` token
  domain, so the first genuine parse may legitimately refuse. Preserve the
  refusal; do not widen the grammar silently.
