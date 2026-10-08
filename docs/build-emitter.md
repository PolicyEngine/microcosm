# Local build publishing service

`microcosm.build.emitter_service` is one local process with a private Unix socket
and independently scheduled publishing components. Its runtime does not know
about event schemas, graph documents, collector registration, or upload APIs.

The composition entrypoint installs two components:

| Component | Responsibility | Failure and retention behavior |
| --- | --- | --- |
| Telemetry | Resource sampling, heartbeats, progress events, collector registration and delivery | Existing best-effort policy; event retention and missing-credential rules remain unchanged |
| Orrery publication | Upload an exact preserved graph/evidence inventory and finalize its immutable publication | Independent durable jobs, bounded exponential retry, and explicit retry command; no telemetry registration or event pruning |

Each component runs in its own worker. Slow graph I/O cannot delay heartbeat
sampling or local acknowledgements for progress events. If a worker raises, the
runtime restarts that worker without stopping other components. The service
checks the build process's lifetime and notifies every component when it exits.
Shutdown is bounded; unfinished graph jobs retain their bytes and database lease
and become eligible again after the lease expires.

The shared `CollectorSession` owns short-lived authentication, credential
refresh, and synchronization between workers. Queue policies stay in the
individual publishers. It obtains the existing HF credential and exchanges it
with the collector; neither component receives a Vercel storage credential.

## Add another publisher

Implement `EmitterComponent` with a unique `name` and these methods:

- `start()` initializes that component's local state.
- `handle(message)` returns true only for an action it owns, after durable enqueue.
- `run(stop)` performs delivery and observes the shared stop event.
- `parent_exited()` records or preserves component-specific state.

Add the instance to the composition entrypoint's component list. Do not add a
publisher-specific branch to the socket runtime or put new jobs into the event
queue merely to reuse its delivery loop. Keep independently meaningful retry
and retention policies separate. A component may use the shared session but must
not impose its rejection policy on another component's jobs.

The service can start without `--registration-json` for graph-only publishing.
The explicit `microcosm-publish-graph` retry command likewise requires only a
publication ID and its preserved inventory, not a run ID. Existing
`LocalTelemetryEmitter` imports remain compatible; the general public handle is
also exported as `LocalBuildEmitter` by `microcosm.build.emitter`.

The SQLite location and the original telemetry package's database/migration
modules retain their existing names to preserve #1099 compatibility. They are
storage compatibility details, not the service's architectural boundary.
Graph jobs have their own table without a telemetry-run foreign key.

## Verification

Engine-free tests exercise a third synthetic component without changing the
runtime, a blocked graph worker alongside a responsive publisher/socket, shared
authentication, graph-only startup, queue persistence across restart, telemetry
pruning, missing/rejected credentials, exact-byte upload, and explicit retry.
Producer HTTP tests use fake responses and never import the runs app.
