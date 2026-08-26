# Logbook archives

Every build attempt -- successful, failed, refused, or discarded -- records a
hash-chained row (microcosm#628, adoption in #666). This directory holds the
public git archives of those rows.

## Layout

```
logbook/us.jsonl                    # grandfathered mixed US chain
logbook/<country>/<dataset>.jsonl   # one new hash chain per dataset line
logbook/families/<scope>.jsonl      # family identity and source checksum
logbook/family_members/<scope>.jsonl # family-to-build associations
logbook/family_actions/<scope>.jsonl # revocations and replacements
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
its spine, staging, and every imputation stage share one chain), `local` is
the local-areas product line, and a future firm line will carry its own
base-data name. Different base data need different scopes, and scope is also
where builds serialize: if two lines must append concurrently, they need
separate chains.

The three family-data directories are not build sequences. Their files use
the same scope paths for validation and selection, but their records have no
predecessor checksum. Build validation and rendering explicitly skip these
directories.

## Build record versions and dataset families

Committed build records created before typed cardinality support retain their
original JSON fields and checksum. Current writers use
`row_format_version = 2`. That representation adds `requested_k`,
`realized_k`, and `record_unit` to the checksum input. A full-size request such
as `N` is resolved before recording, so the stored values are positive numbers;
for example, a full input with 100,000 households records 100000 in both
cardinality columns and `household` as the unit. A failed build may retain a
known request and leave the realized value null.

The existing `rung` field stores a sampling-fraction category such as `f010`.
An exact-k request specifies an absolute household count, not a fraction, so
the exact-k launcher writes SQL/JSON null in `rung`. Version-2 validation
allows that null value and includes it in the build checksum. Legacy rows and
current fraction-based writers continue to require one of the established
fraction values.

The relational model has three parts:

- `families` assigns a caller-created UUID to one Logbook scope and one
  verified prepared-input manifest checksum. The UUID and checksum are
  separate: the UUID is the database identifier, while the checksum describes
  the input used by the family.
- `family_members` contains only `family_id` and `build_id`. It states that the
  build used the family's prepared input. Dataset properties such as size,
  random seed, file location, and build status remain on the build record.
- `family_actions` records either `revokes`, which withdraws one family member,
  or `supersedes`, which says one same-size family member directly replaces
  another. Revocation and replacement remain separate facts.

A build may belong to at most one family. A family may contain any number of
builds. These associations do not change the per-scope predecessor sequence,
and there is no additional release entity or general relationship graph.

## The vocabulary is closed-world

The ratified scopes are exactly `us`, `uk/frs`, and `uk/local` — deliberately
minimal: a scope is ratified when its line starts archiving, not speculatively.
The `uk/local` scope was ratified for the local-areas line's first archived run
in #761. A pipeline whose name derives any unratified scope is refused — even
at genesis. Opening a scope is a reviewed decision made in three places
together: the
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

Family records are exported from the durable spool into all three files for
one scope:

```bash
python tools/logbook.py family-export --scope us --source <run-dir>/logbook-spool
python tools/logbook.py family-export --scope uk/frs --remote
```

To restore archives, first copy the scope's build archive and three family
archives into a local spool, then send the spool. The import rejects a member
whose archived build belongs to another scope. Reconciliation sends queued
builds first, families second, memberships third, and actions last. If a
request fails, its file and every dependent file remain available for the same
command to retry.

```bash
python tools/logbook.py family-import --scope us --spool logbook-spool
python tools/logbook.py reconcile --spool logbook-spool
```

The exact-k launcher uses configuration format version 2 and requires the
caller to create the family UUID:

```json
{
  "schema_version": 2,
  "family": {"id": "12345678-1234-4234-9234-123456789abc"},
  "pool": {
    "release_id": "prepared-pool-release",
    "manifest_sha256": "<verified lowercase SHA-256>"
  }
}
```

Use the same `family.id` for every exact-k build that uses that prepared input.
The launcher verifies `pool.manifest_sha256`, stores that same checksum as the
family's `source_pool_sha256`, resolves `N` to a numeric household count, and
writes the build, family, and membership under `<out>/logbook-spool/`. It then
attempts remote insertion in dependency order. If credentials or database
access are unavailable, all files remain local; run `tools/logbook.py
reconcile --spool <out>/logbook-spool` later. A family UUID already associated
with another source checksum is rejected, and the conflicting local files are
retained for inspection.

Pass the current US predecessor checksum with
`--logbook-prev-row-digest` or `POPULACE_LOGBOOK_PREV_ROW_DIGEST`. The exact-k
launcher still only writes a package and a manual publication command; it does
not publish, certify, or update a published-release pointer.

The archive queries emit compact JSON, suitable for reading directly or
piping to `jq`:

```bash
python tools/logbook.py list-families
python tools/logbook.py list-family-builds --family-id <uuid>
python tools/logbook.py show-family-history --family-id <uuid>
```

This implementation completes the typed exact-count fields requested by
issue #641. It supplies the relational family, membership, and action storage
needed by issue #637; workflow-specific discovery and user interfaces remain
separate work.

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
`uk-frs-staging` or `uk-local-rowwise` to `uk/frs` and `uk/local`;
`logbook.scope_declared` then refuses any scope outside the ratified list. The per-scope advisory lock lets independent scopes append
concurrently while appends within one scope still serialize.

Remote export reads one scope at a time. For `logbook/us.jsonl`, it requests
only the grandfathered US pipelines; for `logbook/<country>/<dataset>.jsonl`,
it requests matching `<country>-<dataset>-*` pipelines and then verifies the
scope again client-side before ordering the chain.

Family export also reads one scope at a time. It filters `families`, the public
member-build view, and the public action view by the stored family scope, then
validates all family and member references before changing an archive.

## Database migration boundary

The family migration must be applied only after the base Logbook, prediction,
fraction-category, and `20260818000000_logbook_chain_scopes.sql` migrations.
Immediately before deployment, the database owner privately identifies the
target and compares its migration history with this order. The repository does
not record the Supabase organization, project name, or project reference. This
repository change neither inspects the production database nor deploys the
migration; both are separate, explicitly authorized operations.

## UK M1 receipt

The campaign's M1 row remains a local receipt only. It is structurally
un-insertable into the database chain, so the `uk/frs` DB chain opens at the
campaign's M2 genesis, matching the archive.
