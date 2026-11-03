# UK graph publication

UK full, dense, size and national builds export `graph.orrery.json` from the
saved graph schema and recorded execution evidence. They preserve the explicit
allowlisted file inventory in the attempt directory, attempt publication before
Hugging Face artifact staging when `--publish-orrery` asks for it, and register
graph/evidence files and an `orrery.publication.json` receipt snapshot in the
artifact manifest.

When execution was recorded, they also write the execution overlay that
`graph.orrery.json` embeds as `graph.execution.json` and register it in the
same manifest. A failed build's bundle carries it too. It is not copied to the
preserved `orrery-publication/` directory or listed in `orrery.upload.json`:
the emitter's file roles and the runs service's contract do not admit it yet.
Its content is already inside `graph.orrery.json`, so registering it discloses
nothing new.

Microcosm's build adapter (`microcosm.build.telemetry_emitter`) starts the
separately installed `microcosm-emitter` command as the build's child process
through `microcosm-emitter-interface`, with independent telemetry, graph
publication and Hugging Face hand-over modules. The emitter owns sockets,
authentication, persistence, retries and resource sampling. Microcosm retains
graph generation, public-file selection and the staged dataset upload.
Slow uploads cannot delay socket acknowledgement or telemetry delivery. A
command that started no emitter of its own starts a short-lived one with only
the graph module to measure and queue the files.

UK full-build commands explicitly defer process shutdown when recording a completed,
validation-refused or failed telemetry outcome. Their final cleanup closes the combined
process after graph inventory and enqueue requests, so failure evidence uses the same
graph socket. Other callers close the process automatically when reporting their outcome.

Publication IDs do not depend on telemetry run IDs. Jobs use SQLAlchemy ORM and
Alembic inside the emitter service's existing local SQLite database. Pending jobs survive missing or
rejected credentials, process restarts and telemetry event pruning. Retry
leases prevent a stale worker from replacing another worker's result.
The exact source bytes must remain available until delivery completes.

The authenticated API creates upload permissions, receives exact files using
scoped signed URLs, then finalizes an immutable publication. Signed uploads
never receive the collector bearer credential. The API rejects different bytes
under an existing publication ID. No population H5 or content-store arrays are
included; only allowlisted graph declarations, execution receipts and reviewed
aggregate summaries can be published publicly.

## Configuration

| Setting | Source and consumer | Provisioning |
| --- | --- | --- |
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | Existing operator credential, read by the emitter from the environment it inherits from the build and exchanged for a short-lived collector session | Existing environment or Hugging Face login; never committed |
| `MICROCOSM_EMITTER_COMMAND` | Optional path of the `microcosm-emitter` command, read by the interface in the build; without it the interface searches `PATH` | Operator's shell; the Modal image sets it |
| Collector origin | Built into the installed emitter; the build sends no address unless a caller passes a loopback development address in the module settings | No producer variable or destination override |
| Runs origin | Tracked `DEFAULT_GRAPH_PUBLICATION_ORIGIN` | No new producer variable |
| `XDG_CACHE_HOME` | Optional existing cache-location setting, read by the interface in the build; the build passes the spool path (`microcosm/telemetry/events.sqlite3`) and the hand-over database beside it (`events.handovers.sqlite3`) to the emitter on its standard input | Defaults to the user's cache directory |

The separately deployed runs application retains its Vercel Blob configuration.
Authenticated production publication also requires
PolicyEngine/calibration-diagnostics#208 deployed, and the runs application's
publication API. Producer tests do not contact either hosted service.

No new secret is required. Tests isolate ambient credentials and caches,
replace the interface's emitter client with an in-memory one, and refuse to
start the installed command.

A build exports locally without remote submission unless `--publish-orrery`
enables it (María's ruling of 2026-10-26), whatever its staging:
`--no-publish-orrery` states the default, and `--publish-orrery` submits even
when dataset staging is local-only. A publication failure does not replace
the dataset's validation result or prevent HF artifact staging.

Since the same ruling the build no longer withholds the figures of target rows
resting on fewer than ten source households, or the per-area support figures
in the gates' failure lines, so the summaries never carry them:

- a calibration summary keeps each target row's target, tolerance and source,
  but no row's estimates or errors, and the size summary keeps the losses but
  not the largest change in one row;
- a gate summary keeps each gate's verdict and its count of failure lines, but
  not the lines.

The build's private diagnostics and gate reports keep all of them. Before it
preserves anything for publication, the export checks the summaries and stops,
with the `unpublishable_summary` error code, if one carries a row's estimates
or a failure line. With no `orrery.upload.json` inventory written, neither the
build nor the retry command below can submit them (Vahid's review of
microcosm#1396).

Failed builds preserve recorded graph evidence before temporary-file cleanup.
They publish it, with an `orrery.failure.json` marker, to `orrery-failure/` in
the attempt evidence directory named by `failure.json`, and leave the output
directory, including an earlier successful bundle, unchanged. An interrupted
build preserves the same local evidence and queue entry but does not wait for
an upload or stage to Hugging Face.

## Retry

```bash
microcosm-emitter-publish-graph --publication-id PUBLICATION_ID \
  --directory /path/to/attempt/orrery-publication
```

The preserved directory contains `orrery.upload.json` and the exact files.
Retries update a separate `publication.status.json`, never the original
HF receipt snapshot. Pending jobs also resume when a subsequent emitter
starts. The retry command is installed with the emitter; Microcosm does not
retain a second delivery implementation or a compatibility wrapper.

## Emitter dependency

`packages/microcosm-build` depends on `microcosm-emitter-interface`
(`>=0.7.0,<1`), the dependency-free package of typed calls, which `uv.lock`
records with its hashes. The emitter itself, `microcosm-emitter`, is not a
workspace dependency: each machine installs it on its own with
`uv tool install microcosm-emitter`, and the Modal image installs the newest
release when it is built for a plan commit. A running build keeps the
emitter it started; the next build uses whichever release is installed then.
So an emitter release needs no Microcosm change. The socket protocol carries
its own version (`PROTOCOL_VERSION`), every emitter release accepts the
versions interfaces still send, and an emitter that lacks a module the build
asks for starts without it.

To adopt a new interface release, regenerate `uv.lock`, update
`APPROVED_UV_LOCK_SHA256` in `microcosm.build.us_runtime.worker_identity` to
the new lockfile digest and rerun the tests and wheel checks.

## Migrating an existing workstation

Install the emitter once with `uv tool install microcosm-emitter` (0.6.0 or
later; a build does not start an older emitter, says how to upgrade it and
continues without it). It keeps using the existing spool database,
`$XDG_CACHE_HOME/microcosm/telemetry/events.sqlite3`, at the same Alembic
revision the pinned 0.2.1 release used, so queued telemetry and graph jobs
keep their IDs and eligibility; do not delete a queue to migrate. The
Hugging Face module keeps its hand-overs in its own SQLite file beside it,
`events.handovers.sqlite3`, so emitters of every release can share the spool
database on one machine. The emitter rejects unversioned databases without
modification. The emitter's README describes the repair for a machine that
ran an earlier, unreleased build of the hand-over queue. No new credential,
hosted service or infrastructure is required by this migration.
