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
names the line's base data, not an epic and not a build mechanism: `frs` is
the FRS-derived survey-microdata line (persons, benunits, and households —
its spine, staging, and every imputation stage share one chain), `locals` is
the local-areas product line, and a future firm line will carry its own
base-data name. Different base data need different scopes, and scope is also
where builds serialize: if two lines must append concurrently, they need
separate chains.

## The vocabulary is closed-world

The ratified scopes are exactly `us` and `uk/frs` — deliberately minimal: a
scope is ratified when its line starts archiving, not speculatively. The
`uk-locals-*` drivers already follow the naming convention, and their rows
spool locally regardless (recording never consults this list); `uk/locals`
gets ratified when the local-areas line first archives. A pipeline whose
name derives any unratified scope is refused — even at genesis. Opening a
scope is a reviewed decision made in three places together: the
`logbook.scope_declared` migration, the `DECLARED_SCOPES` mirror in
`tools/logbook.py`, and this list. It is never a side effect of a
well-formed pipeline name: without the list, a typo'd `uk-huseholds-...`
run would mint scope `uk/huseholds` at genesis, and on an append-only store
a stray scope can never be removed.

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
python tools/logbook.py export --archive logbook/uk/frs.jsonl --source <run-dir>/logbook-spool
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
`uk-frs-staging` or `uk-locals-rowwise` to `uk/frs` and `uk/locals`;
`logbook.scope_declared` then refuses any scope outside the ratified list. The per-scope advisory lock lets independent scopes append
concurrently while appends within one scope still serialize.

Remote export reads one scope at a time. For `logbook/us.jsonl`, it requests
only the grandfathered US pipelines; for `logbook/<country>/<dataset>.jsonl`,
it requests matching `<country>-<dataset>-*` pipelines and then verifies the
scope again client-side before ordering the chain.

## UK M1 receipt

The campaign's M1 row remains a local receipt only. It is structurally
un-insertable into the database chain, so the `uk/frs` DB chain opens at the
campaign's M2 genesis, matching the archive.
