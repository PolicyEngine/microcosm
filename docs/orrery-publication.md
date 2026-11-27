# UK graph inputs

UK full, dense, size and national builds save their graph's compiler schema
(`graph.schema.json`) and, when execution was recorded, its native execution
evidence (`execution.evidence.json` and the `evidence-*.json` files it
indexes). The build's graph step
(`microcosm.build.uk_runtime.orrery_inputs.finalize_graph`) then writes the
execution overlay, `graph.execution.json`, beside the schema, and registers
the collector's inputs under `orrery/` in the build's artifact manifest.
Microcosm makes no Orrery document. The collector
(PolicyEngine/microcosm-collector) builds the Orrery graph files from these
inputs with its own copy of the projection, and staff view a build's graph
through the run viewer (microcosm-runs), which reads it from the collector.
[Orrery inputs](orrery-adapter.md) describes what each file holds.

The graph step registers:

| File | Role in the hand-over | Collector role |
| --- | --- | --- |
| `graph.schema.json` | `orrery_schema` | `schema` |
| `graph.execution.json` | `execution_overlay` | `execution_overlay` |
| each `upstream-<sha256>.json` the schema's extensions or the overlay link | `orrery_upstream` | `upstream` |
| `execution.evidence.json` and each recorded run's graph, manifest, binding and summaries files | named after its key, such as `orrery_execution_evidence_json` | not submitted |

The step finds the upstream files by scanning every `url` field in the
schema's `extensions` and in the overlay for an `upstream-<sha256>.json` name,
and checks each file's bytes against the digest in its name. A build that
recorded no execution registers the schema and its upstream files only, with
no overlay. A failed or refused build has no dataset manifest, so the step
registers the same files by name in the build's own evidence manifest,
`orrery.evidence-manifest.json` (`build_kind` `uk_graph_evidence`,
`releasable` false). When the step registers nothing, it writes no such
record and removes one an earlier attempt left in a reused output directory,
so staging never takes that attempt's files for this one's.

`microcosm.build.staging_dataset.output_role` gives the roles above to the
three kinds of input, registered under `orrery/` in a dataset bundle or by
name in a failed build's evidence bundle. The hand-over names the build for
the collector (see [UK staging operations](uk-staging-operations.md#staged-datasets)),
so the emitter submits the inputs as one `orrery-inputs` submission once the
bundle's Hugging Face commit lands. This is unchanged since microcosm#1590.
Every emitter a build or the re-stage command starts has its collector
submissions on (`telemetry_emitter.COLLECTOR_SUBMISSIONS`, from emitter
0.15.0), and the build waits for the collector to accept the bundle's
submissions, its calibration diagnostics and its Orrery inputs alike, as that
section describes. The collector projects the Orrery document from the
schema and the overlay, validates it, and keeps the graph of a build that
used licensed microdata, as every UK build does, private.

A build submits its Orrery inputs only with remote staging: with
`--staging-local-only`, `--no-staging` or `--no-staged-dataset`, or with
`HF_HUB_OFFLINE` set, its graph files stay on disk. Nothing is projected
locally, so a build staged only locally, or interrupted, has no Orrery
document until its inputs are submitted. The re-stage command
(`tools/stage_uk_rowwise_candidate.py`) can stage a finished run's dataset
bundle kept on disk later, Orrery inputs included; it does not stage a failed
build's evidence bundle. For a build whose files exist only on one machine,
`python -m microcosm.build.uk_runtime.orrery_inputs` writes the same inputs
into a new directory and uploads nothing (see
[Writing inputs for a saved build](orrery-adapter.md#writing-inputs-for-a-saved-build));
a later emitter command, `microcosm-emitter backfill-orrery <dir>`, is planned
to run it and submit the result.

Builds have no `--publish-orrery` or `--no-publish-orrery` option any more.
The first asked for an upload to the public Orrery (María's ruling of
2026-10-26 made that upload opt-in), and both changed nothing once builds
stopped uploading their graph; a build given either now stops with an
unrecognized-argument error.

Microcosm's build adapter (`microcosm.build.telemetry_emitter`) starts the
separately installed `microcosm-emitter` command as the build's child process
through `microcosm-emitter-interface`, with two independent modules:
telemetry and the Hugging Face hand-over. It starts no Orrery graph
publication module. The emitter owns sockets, authentication, persistence,
retries and resource sampling. Microcosm keeps the graph's schema, evidence
capture, the overlay and the choice of files to hand over; the staged dataset
bundle, including a failed build's graph evidence, goes to the same emitter's
Hugging Face module.

UK full-build commands keep the emitter running after they record a
completed, validation-refused or failed telemetry outcome
(`close_service_on_finish=False`), so the attempt's later hand-overs reach
the same emitter, and close it in their final cleanup. Other callers close
the emitter when they report their outcome.

Population files and content-store payloads are never among the graph files:
only the graph declaration and schema, digest-bound native evidence and
reviewed aggregate summaries are registered.

## Status

Builds made before microcosm#1590 and run with `--publish-orrery` queued
uploads to the runs application in the emitter's local queue. The runs
application accepts an upload only after the collector authorizes it, and the
collector has no authorization endpoint (PolicyEngine/calibration-diagnostics#208,
which would have added one, was closed unmerged), so none of them can
succeed. Each stays pending in its machine's queue, with its files in the
attempt's preserved `orrery-publication/` directory, until a later emitter
change removes the direct upload and handles them. That directory never holds
an execution overlay: the overlay is in the build's staged bundle
(`graph.execution.json`), where the collector's rebuild of past builds reads
it.

Builds made before microcosm#1745 also wrote `graph.orrery.json`, the
publication inventory `orrery.upload.json` and the receipt snapshot
`orrery.publication.json`, and registered them in their manifests; none of
them is submitted to the collector.

## Configuration

| Setting | Source and consumer | Provisioning |
| --- | --- | --- |
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | Existing operator credential, read by the emitter from the environment it inherits from the build and exchanged for a short-lived collector session | Existing environment or Hugging Face login; never committed |
| `MICROCOSM_EMITTER_COMMAND` | Optional path of the `microcosm-emitter` command, read by the interface in the build; without it the interface searches `PATH` | Operator's shell; the Modal image sets it |
| Collector origin | Built into the installed emitter; the build sends no address unless a caller passes a loopback development address in the module settings | No producer variable or destination override |
| `XDG_CACHE_HOME` | Optional existing cache-location setting, read by the interface in the build; the build passes the path of the emitter's event queue file (`microcosm/telemetry/events.sqlite3`) and the hand-over database beside it (`events.handovers.sqlite3`) to the emitter on its standard input. On Modal, the stage runner's `microcosm-emitter status` reads it, with `HOME`, in the tool's environment after the tool, so it reads the file the build's emitter used | Defaults to the user's cache directory; on Modal, the container's defaults (the runner sets neither variable, and a plan may not) |
| `MICROCOSM_TELEMETRY_REQUIRED` | Optional, read by the build adapter (`microcosm.build.telemetry_emitter`) each time a build starts its emitter; only builds that start the adapter read it. `1`: the build stops with `TelemetryRequiredError` if the emitter cannot be prepared or is missing, its `version` times out or fails, it is older than `MINIMUM_EMITTER_VERSION`, it lists no protocol version 1, its `hello` fails, it exits or times out before it is ready, or it starts without its telemetry module (for a failure the interface reports, its line saying the build will continue does not apply); unset or `0`: it continues without the emitter; any other value: `ValueError` | Whoever starts the build. The Modal stage runner sets `1` unless the plan sets `0`; workstations: the operator's shell. Leaving it off (unset or `0`) exists only for people outside PolicyEngine who run Microcosm without access to the collector; PolicyEngine's own builds keep telemetry required. Not a credential |
| `MICROCOSM_TELEMETRY_RUN_ID` | Optional, read by the US build programs that open a run through `microcosm.build.program_telemetry`: the telemetry run ID they report under; the part before its last `.` becomes the run's candidate ID. Unset, each attempt makes its own ID (`<pipeline>-<UTC time>-<8 hex digits>`); a value the collector would refuse stops the program with `ValueError` before any work | The Modal stage runner sets `<plan run_id>.<stage>`, and a plan may not set it; workstations: the operator's shell, usually unset. Not a credential |
| `MICROCOSM_TELEMETRY_PARENT_PID` | Read by the same programs: when it equals the process's parent's ID, the process is a child stage whose parent records its telemetry, and it starts no emitter | Set by `build_us_puf_support_base.py --stage all` for the child interpreters it starts; a plan may not set it. Not a credential |
| `MICROCOSM_US_BASE_STEP_REPORT` | Read by the step processes of `build_us_puf_support_base.py`: when it names the process's parent, the process runs the 16 steps from `capital_gains_tail_transfer` to `education_inputs_post_clone`, from its `--stage` on, and writes each step's name to the pipe the variable names as the step starts; the parent then records that step's telemetry stage. A malformed value stops the process | Set by `build_us_puf_support_base.py --stage all` for the one child it starts for those steps, and removed from the environment of its other children. Not a credential |

Producer tests contact neither the runs application nor the collector.

No new secret is required. Tests isolate ambient credentials and caches,
replace the interface's emitter client with an in-memory one, and refuse to
start the installed command.

A graph-step failure does not replace the dataset's validation result or
prevent Hugging Face staging of the dataset bundle. The step registers no
graph file and prints a warning naming the error's type but not its text,
which could carry a path to raw input data.

Since María's ruling the build no longer withholds the figures of target rows
resting on fewer than ten source households, or the per-area support figures
in the validation checks' failure lines, so the summaries never carry them:

- a calibration summary keeps each target row's target, tolerance and source,
  but no row's estimates or errors, and the size summary keeps the losses but
  not the largest change in one row;
- a validation-check summary keeps each check's verdict and its count of
  failure lines, but not the lines.

The build's private diagnostics and validation-check reports keep all of
them. Before it writes the overlay, the graph step checks the recorded
summaries (`orrery_contract.public_summary_findings`). If one carries a target
row's figures or a check's failure lines, the step writes no overlay and
registers no graph file, so the bundle hands the collector no Orrery inputs,
and the build prints a warning that it "hands the collector no Orrery inputs"
(Vahid's review of microcosm#1396). The step also checks the schema's
presentation contract (`validate_presentation`), which the collector's
projection would refuse; any failure other than unpublishable summaries
registers no graph file and prints the error's type.

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
(`>=0.18.0,<1`), the dependency-free package of typed calls, which `uv.lock`
records with its hashes. The emitter itself, `microcosm-emitter`, is not a
workspace dependency: each machine installs it on its own with
`uv tool install microcosm-emitter`. The US Modal stage runner's image
installs the newest release when it is built for a plan commit; the UK
launcher's image (`tools/modal_uk_cut.py`), which clones its commit only at
run time, installs the newest release when Modal first builds that step and
keeps it until the minimum below rises or Modal rebuilds the layer, as it
does when an earlier step of the image or Modal's image builder version
changes. A running build keeps the
emitter it started; the next build uses whichever release is installed then.
So an emitter release needs no Microcosm change. The socket protocol carries
its own version (`PROTOCOL_VERSION`), every emitter release accepts the
versions interfaces still send, and an emitter that lacks a module the build
asks for starts without it.

A build does not start an emitter older than `MINIMUM_EMITTER_VERSION` in
`microcosm.build.telemetry_emitter`, now 0.17.0, and both Modal images install
at least the same release (`microcosm-emitter>=0.17.0`): the US stage
runner's through `image_emitter_requirement()` in
`tools/modal_us_stage_plan.py`, the UK launcher's through
`emitter_requirement()` in `tools/modal_uk_cut.py`. The release is
written once, in microcosm-build's package data,
`packages/microcosm-build/src/microcosm/build/minimum_emitter_version.json`:
the adapter reads it as package data, and both launchers, which import
nothing from the workspace, read it by its path in the checkout that
launches the app. At a Modal launch the install step therefore comes from
the launching checkout and the minimum from the commit the build runs. The
US runner's image adds the launching checkout's file beside its copy of the
plan module, so its check, which runs in the container, reads the same file
(see the [US Modal stage runbook](us-modal-stage-runbook.md)). The UK
launcher's plan reads the commit's file with `git show` and refuses a
commit whose minimum is higher than the launching checkout's. Releases
before 0.10.0 send telemetry to the collector's Cloud Run address, which
changes when the collector moves to another Google Cloud project. From 0.10.0
on, the emitter's default collector address is the collector's hostname,
which stays the same. Microcosm passes the emitter no collector address: the
adapter's `development_collector_url` parameter, which no build command sets,
accepts only a loopback address, and reaches telemetry and the Hugging Face
module's collector submissions alike.

The minimum is 0.17.0, the newest release on 2026-11-19. 0.17.0 includes
`parent_build_manifest.json` and `reform_validation.json`, which a release
publication hands over under the roles `parent_build_manifest` and
`reform_validation` when the release has them (only some US releases do), in
the build's `calibration-diagnostics` submission to the collector, so the
collector receives them for a build whose calibration tree it first builds
from that submission. 0.16.0 leaves both out. The minimum before it, 0.16.0,
from 2026-11-16, added collector provenance on a promotion or withdrawal: once
the change lands, it submits `releases.json` at the commit its receipt reports
(for a change that writes nothing, the commit that last changed the file), and
the collector points the country's latest calibration tree at its default
release's current build when it has that build's tree. 0.15.0 accepts that
provenance but drops it, so it would submit no `releases.json`. The minimum
before that, 0.15.0, from 2026-11-13, added the Hugging Face module's setting
that switches collector submissions on, which every emitter a build starts
sets (`telemetry_emitter.COLLECTOR_SUBMISSIONS`), collector provenance on a
release publication, which it submits after the release commit, and a
`skipped` collector receipt with an error code when it refuses its collector
settings. An emitter before 0.15.0 accepts the setting but drops a
publication's provenance, so it would submit no release. The minimum before
that, 0.14.0, added the build machine's readings to each heartbeat, the sleeps
the machine took, a description of the machine in the run's first event, each
stage's awake and asleep time at its end, and a counter one level above a
stage's progress, which the US fiscal refresh's post-export scoring sends as
the reform pass each scoring batch belongs to (the interface's
`OuterProgress`, from 0.14.0). 0.13.0 added `microcosm-emitter status`, which
the US Modal stage runner runs after each stage to record how many telemetry
events the stage's emitter left undelivered; builds require the same release
as the Modal image so that it is written once. Practically no machine ran the
separately installed emitter on any of these dates, so refusing 0.11.9 to
0.16.0 cost nothing; the minimum before those, 0.11.9, came from the
collector's hostname, for which 0.10.0 would have been enough. A refused
emitter does cost the build:

- no hosted telemetry, and no staging hand-overs, including the Orrery inputs
  in a UK build's staged bundle (the staging run files and the graph files are
  still written locally);
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

Install the emitter once with `uv tool install microcosm-emitter` (0.17.0 or
later; a build does not start an older emitter, says how to upgrade it and
continues without it). It keeps using the emitter's existing event queue
file, `$XDG_CACHE_HOME/microcosm/telemetry/events.sqlite3`, at the same
Alembic revision the pinned 0.2.1 release used, so queued telemetry events,
and the graph uploads that builds before microcosm#1590 queued (see
[Status](#status)), keep their IDs and eligibility; do not delete a queue to
migrate. The Hugging Face module keeps its hand-overs in its own SQLite file
beside it, `events.handovers.sqlite3`, so emitters of every release can share
the event queue file on one machine. The emitter rejects unversioned
databases without modification. The emitter's README describes the repair
for a machine that ran an earlier, unreleased build of the hand-over queue.
No new credential, hosted service or infrastructure is required by this
migration.

To upgrade a machine whose emitter is older than 0.17.0, run
`uv tool install --upgrade 'microcosm-emitter>=0.17.0'`; it also replaces a
version constraint given at install time, which `uv tool upgrade` keeps.
`microcosm-emitter version` prints the installed release.
