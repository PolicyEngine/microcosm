# UK graph publication

UK full, dense, size and national builds export `graph.orrery.json` from the
saved graph schema and recorded execution evidence. They preserve the explicit
public-safe file inventory in the attempt directory, attempt publication before
Hugging Face artifact staging, and register graph/evidence files and an
`orrery.publication.json` receipt snapshot in the artifact manifest.

When execution was recorded, they also write the execution overlay that
`graph.orrery.json` embeds as `graph.execution.json` and register it in the
same manifest. A failed build's bundle carries it too. It is not copied to the
preserved `orrery-publication/` directory or listed in `orrery.upload.json`:
the emitter's file roles and the runs service's contract do not admit it yet.
Its content is already inside `graph.orrery.json`, so registering it discloses
nothing new.

Microcosm's metadata adapter starts one local emitter service with two independent
workers: live telemetry and completed graph publication. The `microcosm-emitter` package
owns sockets, authentication, persistence, retries and resource sampling.
Microcosm retains graph generation, public-file selection and HF staging.
Slow uploads cannot delay socket acknowledgement or telemetry delivery.

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

## Status

No upload is published today. The runs application accepts an upload only
after the collector authorizes it, and the collector has no authorization
endpoint: PolicyEngine/calibration-diagnostics#208, which would have added
one, was closed unmerged. Builds still queue uploads as described above. A
queued job stays pending in the emitter's local queue, its files stay in the
preserved directory, and no retry can succeed.

The collector will publish Orrery graph files itself instead. Builds will
submit the files to the collector's artifact intake
(PolicyEngine/microcosm-collector#12), and the collector's graph worker will
build and publish the Orrery graph files from them. The direct upload to the
runs application will then be removed.

By default, a build queues an upload only when it also stages its artifact
bundle to Hugging Face, under `staged/<run_id>/`. That bundle holds the same
files, so the Orrery graph files can be rebuilt from it later. This does not
hold for an interrupted build, a build run with `--publish-orrery` without
remote staging, or a build whose Hugging Face staging failed.

## Configuration

| Setting | Source and consumer | Provisioning |
| --- | --- | --- |
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | Existing operator credential, exchanged by the service for a short-lived collector session | Existing environment or Hugging Face login; never committed |
| Collector origin | Tracked `PRODUCTION_COLLECTOR_URL` in the emitter's internal core library | No producer variable or destination override |
| Runs origin | Tracked `DEFAULT_GRAPH_PUBLICATION_ORIGIN` | No new producer variable |
| `XDG_CACHE_HOME` | Optional existing cache-location setting | Defaults to the user's cache directory |

The separately deployed runs application retains its Vercel Blob configuration.
Producer tests do not contact the runs application or the collector.

No new Microcosm runtime variable or secret is required. Tests isolate ambient
credentials and caches and use fake HTTP or explicit loopback services.

`--staging-local-only`, `--no-staging` and `--no-staged-dataset` export locally
without remote submission. `--publish-orrery` explicitly enables submission;
`--no-publish-orrery` disables it independently. A publication failure does not
replace the dataset's validation result or prevent HF artifact staging.

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
HF receipt snapshot. Pending jobs also resume when a subsequent local service
starts. The retry command comes from the emitter package; Microcosm does not
retain a second delivery implementation or a compatibility wrapper.

## Emitter dependency

`packages/microcosm-build` depends on one exact `microcosm-emitter` release
from PyPI, which `uv.lock` records with its wheel and source-archive hashes.
Microcosm has no Git source override for the emitter. Every merge to the
emitter's `main` publishes a new version automatically. To adopt one, change the
exact requirement, regenerate `uv.lock`, update `APPROVED_UV_LOCK_SHA256` in
`microcosm.build.us_runtime.worker_identity` to the new lockfile digest and
rerun the tests and wheel checks.

## Migrating an existing workstation

The emitter rejects unversioned databases without modification. Before switching
an existing workstation, upgrade any old unversioned queue using the preceding
Microcosm implementation. Versioned telemetry and graph jobs keep their existing
IDs, eligibility, path and Alembic revision; do not delete a queue to migrate.
No new runtime variable, credential, hosted service or infrastructure is required
by this migration.
