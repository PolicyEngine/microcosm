# UK graph publication

UK full, dense, size and national builds export `graph.orrery.json` from the
saved graph schema and recorded execution evidence, for viewing the graph
offline. When execution was recorded, they also write the execution overlay
that `graph.orrery.json` embeds as `graph.execution.json`. They register the
graph and evidence files, the emitter's publication inventory
(`orrery.upload.json`) and an `orrery.publication.json` receipt snapshot in
the artifact manifest, and a failed build's evidence bundle carries the same
files.

No build uploads its graph. The collector builds the Orrery graph files
itself, from the inputs the build's staged bundle submits:

| Bundle file | Role in the hand-over | Collector role |
| --- | --- | --- |
| `graph.schema.json` | `orrery_schema` | `schema` |
| `graph.execution.json` | `execution_overlay` | `execution_overlay` |
| each `upstream-<sha256>.json` the projection links | `orrery_upstream` | `upstream` |

`microcosm.build.staging_dataset.output_role` gives these roles to the three
kinds of file, registered under `orrery/` in a dataset bundle or by name in a
failed build's evidence bundle. The hand-over names the build for the
collector (see [UK staging operations](uk-staging-operations.md#staged-datasets)),
so the emitter submits the files as one `orrery-inputs` submission once the
bundle's Hugging Face commit lands. Every emitter a build or the re-stage
command starts has its collector submissions on
(`telemetry_emitter.COLLECTOR_SUBMISSIONS`, from emitter 0.15.0), and the
build waits for the collector to accept the bundle's submissions, its
calibration diagnostics and its Orrery inputs alike, as that section
describes.
The collector projects the document with
`orrery_json_from_schema(schema, execution=overlay)`, validates it and keeps
the graph of a build that used licensed microdata, as every UK build does,
private; staff are to see it through the run viewer once it reads graphs
from the collector. The native evidence
files (`execution.evidence.json` and `evidence-<phase>-*.json`) and
`graph.orrery.json` keep roles named after their keys, so they are not
submitted.

A build submits its Orrery inputs only with remote staging: with
`--staging-local-only`, `--no-staging` or `--no-staged-dataset` its graph
files stay on disk. `--publish-orrery`, which asked for an upload to the
public Orrery (María's ruling of 2026-10-26 made that upload opt-in), now
prints a warning and changes nothing; `--no-publish-orrery` is accepted too.

Microcosm's build adapter (`microcosm.build.telemetry_emitter`) starts the
separately installed `microcosm-emitter` command as the build's child process
through `microcosm-emitter-interface`, with independent telemetry, graph
publication and Hugging Face hand-over modules. The emitter owns sockets,
authentication, persistence, retries and resource sampling. Microcosm retains
graph generation and public-file selection; the staged dataset bundle,
including a failed build's graph evidence, goes to the same emitter's Hugging
Face module. The build still asks the emitter's graph publication module for
the publication inventory it registers, and queues nothing with it; the
inventory and the receipt go once the projection code leaves Microcosm. A
command that started no emitter of its own starts a short-lived one with only
the graph module to measure the files.

UK full-build commands explicitly defer process shutdown when recording a completed,
validation-refused or failed telemetry outcome. Their final cleanup closes the combined
process after the graph inventory requests, so failure evidence uses the same
graph socket. Other callers close the process automatically when they report their outcome.

No population H5 or content-store arrays are among the graph files; only
allowlisted graph declarations, execution receipts and reviewed aggregate
summaries are exported.

## Status

Builds made before this change, run with `--publish-orrery`, queued uploads
to the runs application in the emitter's local queue. The runs application
accepts an upload only after the collector authorizes it, and the collector
has no authorization endpoint
(PolicyEngine/calibration-diagnostics#208, which would have added one, was
closed unmerged), so none of them can succeed. Each stays pending in its
machine's queue with its files in the attempt's preserved
`orrery-publication/` directory, until an emitter change removes the direct
upload. That change can re-submit a queued upload to the collector as an
`orrery-inputs` submission only with an execution overlay, which the
preserved directory never holds: the overlay is in the build's staged bundle
(`graph.execution.json`), and the collector's rebuild of past builds reads it
there.

The receipt snapshot (`orrery.publication.json`) of every build now records
`skipped` with the error code `local_only`: nothing was queued for the runs
application. It records `failed` with `local_service_unavailable` when the
build could not measure the inventory, and with `unpublishable_summary` (see
below) when a summary failed its check.

## Configuration

| Setting | Source and consumer | Provisioning |
| --- | --- | --- |
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | Existing operator credential, read by the emitter from the environment it inherits from the build and exchanged for a short-lived collector session | Existing environment or Hugging Face login; never committed |
| `MICROCOSM_EMITTER_COMMAND` | Optional path of the `microcosm-emitter` command, read by the interface in the build; without it the interface searches `PATH` | Operator's shell; the Modal image sets it |
| Collector origin | Built into the installed emitter; the build sends no address unless a caller passes a loopback development address in the module settings | No producer variable or destination override |
| Runs origin | Tracked `DEFAULT_GRAPH_PUBLICATION_ORIGIN` | No new producer variable |
| `XDG_CACHE_HOME` | Optional existing cache-location setting, read by the interface in the build; the build passes the path of the emitter's event queue file (`microcosm/telemetry/events.sqlite3`) and the hand-over database beside it (`events.handovers.sqlite3`) to the emitter on its standard input. On Modal, the stage runner's `microcosm-emitter status` reads it, with `HOME`, in the tool's environment after the tool, so it reads the file the build's emitter used | Defaults to the user's cache directory; on Modal, the container's defaults (the runner sets neither variable, and a plan may not) |
| `MICROCOSM_TELEMETRY_REQUIRED` | Optional, read by the build adapter (`microcosm.build.telemetry_emitter`) each time a build starts its emitter; only builds that start the adapter read it. `1`: the build stops with `TelemetryRequiredError` if the emitter cannot be prepared or is missing, its `version` times out or fails, it is older than `MINIMUM_EMITTER_VERSION`, it lists no protocol version 1, its `hello` fails, it exits or times out before it is ready, or it starts without its telemetry module (for a failure the interface reports, its line saying the build will continue does not apply); unset or `0`: it continues without the emitter; any other value: `ValueError` | Whoever starts the build. The Modal stage runner sets `1` unless the plan sets `0`; workstations: the operator's shell. Leaving it off (unset or `0`) exists only for people outside PolicyEngine who run Microcosm without access to the collector; PolicyEngine's own builds keep telemetry required. Not a credential |
| `MICROCOSM_TELEMETRY_RUN_ID` | Optional, read by the US build programs that open a run through `microcosm.build.program_telemetry`: the telemetry run ID they report under; the part before its last `.` becomes the run's candidate ID. Unset, each attempt makes its own ID (`<pipeline>-<UTC time>-<8 hex digits>`); a value the collector would refuse stops the program with `ValueError` before any work | The Modal stage runner sets `<plan run_id>.<stage>`, and a plan may not set it; workstations: the operator's shell, usually unset. Not a credential |
| `MICROCOSM_TELEMETRY_PARENT_PID` | Read by the same programs: when it equals the process's parent's ID, the process is a child stage whose parent records its telemetry, and it starts no emitter | Set by `build_us_puf_support_base.py --stage all` for the child interpreters it starts; a plan may not set it. Not a credential |

The separately deployed runs application retains its Vercel Blob configuration.
Producer tests do not contact the runs application or the collector.

No new secret is required. Tests isolate ambient credentials and caches,
replace the interface's emitter client with an in-memory one, and refuse to
start the installed command.

A graph export failure does not replace the dataset's validation result or
prevent Hugging Face artifact staging.

Since the same ruling the build no longer withholds the figures of target rows
resting on fewer than ten source households, or the per-area support figures
in the gates' failure lines, so the summaries never carry them:

- a calibration summary keeps each target row's target, tolerance and source,
  but no row's estimates or errors, and the size summary keeps the losses but
  not the largest change in one row;
- a gate summary keeps each gate's verdict and its count of failure lines, but
  not the lines.

The build's private diagnostics and gate reports keep all of them. Before it
preserves anything, the export checks the summaries and stops, with the
`unpublishable_summary` error code, if one carries a row's estimates or a
failure line. It then writes no execution overlay and no
`orrery.upload.json` inventory, so the bundle carries no Orrery inputs for
the collector (Vahid's review of microcosm#1396).

Failed builds preserve recorded graph evidence before temporary-file cleanup.
They publish it, with an `orrery.failure.json` marker, to `orrery-failure/` in
the attempt evidence directory named by `failure.json`, and leave the output
directory, including an earlier successful bundle, unchanged. Its evidence
bundle is staged as a dataset bundle is, naming the build for the collector
with no release, so its Orrery inputs are submitted the same way. An
interrupted build preserves the same local evidence but does not stage to
Hugging Face.

## Emitter dependency

`packages/microcosm-build` depends on `microcosm-emitter-interface`
(`>=0.15.0,<1`), the dependency-free package of typed calls, which `uv.lock`
records with its hashes. The emitter itself, `microcosm-emitter`, is not a
workspace dependency: each machine installs it on its own with
`uv tool install microcosm-emitter`, and the Modal image installs the newest
release when it is built for a plan commit. A running build keeps the
emitter it started; the next build uses whichever release is installed then.
So an emitter release needs no Microcosm change. The socket protocol carries
its own version (`PROTOCOL_VERSION`), every emitter release accepts the
versions interfaces still send, and an emitter that lacks a module the build
asks for starts without it.

A build does not start an emitter older than `MINIMUM_EMITTER_VERSION` in
`microcosm.build.telemetry_emitter`, now 0.15.0, and the Modal image installs
at least the same release (`image_emitter_requirement()` in
`tools/modal_us_stage_plan.py`, `microcosm-emitter>=0.15.0`). The release is
written once, in microcosm-build's package data,
`packages/microcosm-build/src/microcosm/build/minimum_emitter_version.json`:
the adapter reads it as package data, and the runner, which imports nothing
from the workspace, reads it by its path in the checkout that launches the
app. The image adds that checkout's file beside the runner's copy of the
plan module, so the check, which runs in the container, reads the same
file. At a Modal launch the install step therefore comes from the launching
checkout and the minimum from the commit the plan pins (see the
[US Modal stage runbook](us-modal-stage-runbook.md)). Releases
before 0.10.0 send telemetry to the collector's Cloud Run address, which
changes when the collector moves to another Google Cloud project. From 0.10.0
on, the emitter's default collector address is the collector's hostname,
which stays the same. Microcosm passes the emitter no collector address: the
adapter's `development_collector_url` parameter, which no build command sets,
accepts only a loopback address, and reaches telemetry, Orrery graph
publication and the Hugging Face module's collector submissions alike.

The minimum is 0.15.0, the newest release on 2026-11-13. 0.15.0 adds the
Hugging Face module's setting that switches collector submissions on, which
every emitter a build starts sets (`telemetry_emitter.COLLECTOR_SUBMISSIONS`),
collector provenance on a release publication, which it submits after the
release commit, and a `skipped` collector receipt with an error code when it
refuses its collector settings. An older emitter accepts the setting but
drops a publication's provenance, so it would submit no release. The
minimum before it, 0.14.0, added the build machine's readings to each
heartbeat, the sleeps the machine took, a description of the machine in the
run's first event, each stage's awake and asleep time at its end, and a
counter one level above a stage's progress, which the US fiscal refresh's
post-export scoring sends as the reform pass each scoring batch belongs to
(the interface's `OuterProgress`, from 0.14.0). 0.13.0 added
`microcosm-emitter status`, which the US Modal stage runner runs after each
stage to record how many telemetry events the stage's emitter left
undelivered; builds require the same release as the Modal image so that it
is written once. Practically no machine ran the separately installed
emitter on any of these dates, so refusing 0.11.9 to 0.14.0 cost nothing;
the minimum before those, 0.11.9, came from the
collector's hostname, for which 0.10.0 would have been enough. A refused
emitter does cost the build:

- no hosted telemetry, and no staging hand-overs (the staging run files are
  still written locally);
- the UK graph step records a failed receipt (`local_service_unavailable`)
  and writes no publication inventory;
- a UK build given `--staging-read-back` fails at the read-back;
- a UK build that asks for remote staging of its dataset bundle, as UK builds
  do by default, is refused before it starts, and the refusal gives the
  upgrade command built from the minimum.

The build's warning says so. Raising the minimum in that one file raises
the image's floor too; before raising it while machines still run older
emitters, weigh these losses against having each such machine upgrade
first. Raise the installed-emitter job's pin
(`tools/installed-emitter/requirements.txt`) with it, so that the job never
tests a release older than the minimum.

To adopt a new interface release, regenerate `uv.lock`, update
`APPROVED_UV_LOCK_SHA256` in `microcosm.build.us_runtime.worker_identity` to
the new lockfile digest and rerun the tests and wheel checks.

## Migrating an existing workstation

Install the emitter once with `uv tool install microcosm-emitter` (0.15.0 or
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

To upgrade a machine whose emitter is older than 0.15.0, run
`uv tool install --upgrade 'microcosm-emitter>=0.15.0'`; it also replaces a
version constraint given at install time, which `uv tool upgrade` keeps.
`microcosm-emitter version` prints the installed release.
