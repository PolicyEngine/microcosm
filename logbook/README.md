# Logbook archives

Every build attempt — successful, failed, refused, or discarded — records a
hash-chained row (microcosm#628, adoption in #666). This directory holds the
public git archives of those rows.

## Layout

```
logbook/<country>.jsonl        # one hash chain per country
logbook-spool/<country>/       # the committed spool mirror, same shape
```

One file is one chain. Country is the split because the two build
programmes share no build lineage: a UK migration attempt and a US pool
refresh have nothing to say about each other's history, and interleaving
them makes both archives unreadable.

The split stops at country deliberately. Finer scopes — one chain per epic —
would multiply chains as fast as epics appear, and the archive is not where
the epic story belongs: that lives in the archival commit message and the
row's `code_pin`.

## A chain can only be extended

A row's digest is `sha256(canonical non-chain fields ‖ prev_row_digest)`, so
every row commits to its predecessor. That is what makes the archive
tamper-evident, and it has two consequences worth knowing before you build.

**Rows cannot be re-rooted.** A row minted against one chain's tail belongs
to that chain forever; moving it to another archive would mean recomputing
its digest, which is indistinguishable from tampering. `us.jsonl` therefore
stays exactly as it is — it spans three US pipelines (`us-2024-release`,
`us-pool-inc2`, `us-stacked-pool`) because it predates any split, and it
cannot be divided retroactively.

**Concurrent builds in one country fork the chain.** The predecessor is
chosen at build time, so two runs that extend the same tail produce two rows
with the same predecessor — a fork, which export refuses, and neither row
can be re-chained afterwards. Within a country, run builds one at a time, or
accept that a concurrent attempt is spool-only and never archived.

## Operating

Recording happens at build time: drivers spool rows beside the run's output
artifact. Archiving is a separate, reviewed step — the build never writes
into the repository.

```bash
python tools/logbook.py export --archive logbook/uk.jsonl --source <run-dir>/logbook-spool
```

Export verifies the chain before appending and refuses a suffix that does
not continue the archive's current tail. Commit the result as its own
narrative change describing what the attempts established.

Thread runs together by passing the previous row's digest
(`--logbook-prev-row-digest`, or `POPULACE_LOGBOOK_PREV_ROW_DIGEST`) so the
country's chain stays continuous; a country's first archived run roots at
genesis.

Read commands take a single archive or a directory:

```bash
python tools/logbook.py validate            # every chain under logbook/
python tools/logbook.py render              # a section per country
python tools/logbook.py render --archive logbook/us.jsonl
```

`render` is the public-safe projection: it shows `artifact_location` only
for `published` and `certified` rows.

## The live store, and why it does not match yet

The best-effort Supabase insert (`POPULACE_LEDGER_URL` +
`POPULACE_LEDGER_KEY`, the migration's insert-only `logbook_writer` role) is
one table with no country column, and it currently enforces a **single
global chain**: `builds_single_genesis` permits one genesis row table-wide,
`builds_unique_predecessor` forbids forks, and the `enforce_build_chain`
trigger requires every insert to extend the whole table's current tail.

A second country's genesis row is therefore rejected by the live store as
migrated today. Until that project is reorganized to scope chains per
country, these archives are the git-side record and a country beyond the
first stays spool-and-git only. Remote availability was never part of build
correctness — with no credentials a row stays spooled without error — so
nothing about recording changes in the meantime.
