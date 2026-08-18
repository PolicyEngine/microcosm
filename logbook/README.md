# Logbook archives

Every build attempt -- successful, failed, refused, or discarded -- records a
hash-chained row (microcosm#628, adoption in #666). This directory holds the
public git archives of those rows.

## Layout

```
logbook/us.jsonl                    # grandfathered mixed US chain
logbook/<country>/<dataset>.jsonl   # one new hash chain per dataset line
logbook-spool/<country>/<dataset>/  # committed spool mirror, same shape
```

One file is one chain scope. The grandfathered `logbook/us.jsonl` file spans
three US pipelines (`us-2024-release`, `us-pool-inc2`,
`us-stacked-pool`) because it predates this split and its rows already link
across those names. It can never be divided retroactively: the predecessor is
part of each row digest, so moving a row to a different root would recompute
the digest and destroy the evidence the chain exists to preserve.

Every new archive uses `logbook/<country>/<dataset>.jsonl`. The dataset token
is a line of data work, not an epic name: `households`, `locals`, `firms`, and
similar base-data families. Different base data need different scopes, and
scope is also where builds serialize. If two lines must append concurrently,
they need separate chains.

## A chain can only be extended

A row's digest is `sha256(canonical non-chain fields || prev_row_digest)`, so
every row commits to its predecessor. That is what makes the archive
tamper-evident, and it has two consequences worth knowing before you build.

**Rows cannot be re-rooted.** A row minted against one chain's tail belongs
to that chain forever; moving it to another archive would mean recomputing
its digest, which is indistinguishable from tampering.

**Concurrent builds in one scope fork the chain.** The predecessor is chosen
at build time, so two runs that extend the same tail produce two rows with
the same predecessor -- a fork, which export refuses, and neither row can be
re-chained afterwards. Within a scope, run builds one at a time, or accept
that a concurrent attempt is spool-only and never archived.

## Operating

Recording happens at build time: drivers spool rows beside the run's output
artifact. Archiving is a separate, reviewed step -- the build never writes
into the repository.

```bash
python tools/logbook.py export --archive logbook/uk/households.jsonl --source <run-dir>/logbook-spool
```

Export verifies the chain before appending and refuses a suffix that does
not continue the archive's current tail. Commit the result as its own
narrative change describing what the attempts established.

Thread runs together by passing the previous row's digest
(`--logbook-prev-row-digest`, or `POPULACE_LOGBOOK_PREV_ROW_DIGEST`) so the
scope's chain stays continuous; a scope's first archived run roots at
genesis.

Read commands take a single archive or a directory:

```bash
python tools/logbook.py validate            # every chain under logbook/
python tools/logbook.py render              # a section per scope
python tools/logbook.py render --archive logbook/us.jsonl
```

`render` is the public-safe projection: it shows `artifact_location` only
for `published` and `certified` rows.

## The live store

The best-effort Supabase insert (`POPULACE_LEDGER_URL` +
`POPULACE_LEDGER_KEY`, the migration's insert-only `logbook_writer` role)
uses the same scope rule as the archives after this PR's migration is applied
by the project owner. The writer key is unaffected: rows are still inserted
through the same role and the scope is derived from the hashed `pipeline`
field, not supplied as a mutable column.

The database keeps `builds_unique_predecessor` global because two rows
claiming one predecessor is a fork wherever it happens. Genesis, tail
discovery, and advisory locking are scoped: `logbook.chain_scope(pipeline)`
maps the three legacy US pipeline names to `us`, and maps new pipelines like
`uk-households-staging` or `uk-locals-rowwise` to `uk/households` and
`uk/locals`. The per-scope advisory lock lets independent scopes append
concurrently while appends within one scope still serialize.

Remote export reads one scope at a time. For `logbook/us.jsonl`, it requests
only the grandfathered US pipelines; for `logbook/<country>/<dataset>.jsonl`,
it requests matching `<country>-<dataset>-*` pipelines and then verifies the
scope again client-side before ordering the chain.

## UK M1 receipt

The campaign's M1 row remains a local receipt only. It is structurally
un-insertable into the database chain, so the `uk/households` DB chain opens
at the campaign's M2 genesis, matching the archive.
