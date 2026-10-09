# UK graph publication

UK full, dense, size and national builds export `graph.orrery.json` from the
saved graph schema and recorded execution evidence. They preserve the explicit
public-safe file inventory in the attempt directory, attempt publication before
Hugging Face artifact staging, and register graph/evidence files and an
`orrery.publication.json` receipt snapshot in the artifact manifest.

The existing local service has two independent workers: live telemetry and
completed graph publication. Slow uploads cannot delay socket acknowledgement
or telemetry delivery. This change does not introduce another generic runtime,
rename the telemetry package, or add compatibility wrappers.

Publication IDs do not depend on telemetry run IDs. Jobs use SQLAlchemy and
Alembic in the existing local SQLite database. Pending jobs survive missing or
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
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | Existing operator credential, exchanged by the service for a short-lived collector session | Existing environment or Hugging Face login; never committed |
| Collector origin | Tracked `PRODUCTION_COLLECTOR_URL` in the existing service | No producer variable or destination override |
| Runs origin | Tracked `DEFAULT_GRAPH_PUBLICATION_ORIGIN` | No new producer variable |
| `XDG_CACHE_HOME` | Optional existing cache-location setting | Defaults to the user's cache directory |

The separately deployed runs application retains its Vercel Blob configuration.
Authenticated production publication also requires
PolicyEngine/calibration-diagnostics#208 deployed, and the runs application's
publication API. Producer tests do not contact either hosted service.

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
microcosm-publish-graph --publication-id PUBLICATION_ID \
  --directory /path/to/attempt/orrery-publication
```

The preserved directory contains `orrery.upload.json` and the exact files.
Retries update a separate `publication.status.json`, never the original
HF receipt snapshot. Pending jobs also resume when a subsequent local service
starts. A job whose preserved directory has been removed stays pending with
error code `preserved_files_missing` until its files are restored to the same
directory; it does not stop delivery of other jobs. Moving this
service into the `microcosm-emitter` package is a separate, deferred consumer
change; this implementation does not require that package.
