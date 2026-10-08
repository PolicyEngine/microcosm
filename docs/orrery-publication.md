# UK graph publication

UK full, dense, size and national builds export `graph.orrery.json` from the
saved graph schema and recorded execution evidence. They preserve the explicit
public-safe file inventory in the attempt directory, attempt publication before
Hugging Face artifact staging, and register graph/evidence files and an
`orrery.publication.json` receipt snapshot in the artifact manifest.

Microcosm's metadata adapter starts one provider service with two independent
workers: live telemetry and completed graph publication. The provider packages
own sockets, authentication, persistence, retries and resource sampling.
Microcosm retains graph generation, public-file selection and HF staging.
Slow uploads cannot delay socket acknowledgement or telemetry delivery.

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
| Collector origin | Tracked `PRODUCTION_COLLECTOR_URL` in provider core | No producer variable or destination override |
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
microcosm-provider-publish-graph --publication-id PUBLICATION_ID \
  --directory /path/to/attempt/orrery-publication
```

The preserved directory contains `orrery.upload.json` and the exact files.
Retries update a separate `publication.status.json`, never the original
HF receipt snapshot. Pending jobs also resume when a subsequent local service
starts. The retry command comes from the provider package; Microcosm does not
retain a second delivery implementation or a compatibility wrapper.

## Deferred migration prerequisites

This adapter is implemented and tested in a draft PR, not merged or activated.
It depends on PolicyEngine/microcosm-emitter#4 (stacked on #2).
The draft uses exact Git source pins in `packages/microcosm-build/pyproject.toml`
and `uv.lock` because provider version 0.1.0 has not been released by this work.
Before merge, verify registry-owner configuration, release all four Python
packages, remove the draft source overrides, regenerate `uv.lock`, update its
approved worker-identity digest and rerun tests and wheel checks. Merely building
a Microcosm wheel does not make its unpublished dependencies installable.

The provider rejects unversioned databases without modification. Before switching
an existing workstation, upgrade any old unversioned queue using the preceding
Microcosm implementation. Versioned telemetry and graph jobs keep their existing
IDs, eligibility, path and Alembic revision; do not delete a queue to migrate.
No new runtime variable, credential, hosted service or infrastructure is required
by this migration. The previously required hosted authorization/API deployments
remain prerequisites for remote publication.
