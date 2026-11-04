# UK staging operations

This document records the operational ownership and lifecycle decisions for
UK Microcosm staging telemetry. It does not authorize production publication.

## Maintainer responsibilities

- PolicyEngine Hugging Face organization administrators review individual
  access requests and administer `policyengine/populace-uk-staging`. The
  administrator performing initial provisioning becomes the recorded primary
  owner; another organization administrator must be recorded as backup before
  remote telemetry is enabled.
- Administrators of `PolicyEngine/microcosm` administer the GitHub `staging`
  environment. The initial required-reviewer candidates are `@anth-volk` and
  `@MaxGhenis`; environment creation must confirm both accounts still have the
  required repository access before applying that configuration.
- Calibration Diagnostics deployment maintainers administer the server-side
  read credential and deploy consumer support before UK remote output is
  enabled.

The current GitHub repository-administrator list was inspected on 2026-09-08.
It included `@anth-volk`, `@MaxGhenis`, `@nikhilwoodruff`, `@nwoodruff-co`,
`@PavelMakarchuk`, `@policyengine-auto`, and `@vahid-ahmadi`. This dated list is
evidence, not a permanent authorization list; provisioning must query current
permissions again.

The GitHub `staging` environment did not exist when inspected on 2026-09-08.
It was then created with required reviewers `@anth-volk` and `@MaxGhenis`,
self-review prevention enabled, and no environment secrets. The general
integration-test workflow does not reference this protected environment, so
these reviewers do not delay its synthetic tests. The environment is audited
or reconciled idempotently with:

```bash
uv run python tools/configure_github_staging_environment.py
uv run python tools/configure_github_staging_environment.py --apply
```

The setup permits no environment secret except `HF_STAGING_READ_TOKEN`. The
general integration-test workflow instead reads that optional name as a
repository-level Actions secret, where it may inspect the private repository's
settings. Fork pull requests do not receive it. The integration build itself
receives no external service writer credential.

## Contract fixture ownership

Microcosm is the canonical source for staging contract fixtures because it is
the producer and validator of the repository files. Canonical fixtures live
under `packages/microcosm-build/tests/fixtures/staging/` and have a checked-in
SHA-256 manifest. Calibration Diagnostics keeps a byte-identical copy and
checks the same manifest digest. A contract change is incomplete until both
repositories accept the new fixture manifest in their respective feature
branches.

Version 1 fixtures describe current US output and remain fixed during the UK
implementation. Version 2 fixtures (successful spine, calibration, sanitized
failure, delivery-failure, and incompatible-version cases) are frozen: the UK
writer now emits version 3, and the version 2 bytes stay only to prove old runs
still read. Version 3 fixtures add a `blocked` run and a failure that carries
its `failure_class`.

The implementation separates versioned file serialization from country
configuration. `microcosm.build.staging_storage` owns the hand-over of run
files to the build's emitter and its repeated-failure policy, used by both
version 1 and version 2, and the read access that fetches a staged dataset;
`microcosm.build.staging_dataset` hands the staged dataset to the emitter.
The emitter commits every pending run file of a repository in one commit per
upload interval, with the build's own credential, and reads a run back on
request.
`microcosm.build.staging_cli` adds and validates common command-line options
from a supplied repository configuration. The UK repository identifier and
environment-variable name live in `microcosm.build.uk_runtime.staging`; the
shared modules contain no UK repository or environment defaults. The two
serializers remain separate because version 1 and version 2 intentionally
write different files.

Regenerate and verify the canonical version 3 bytes with (the generator no
longer writes version 2):

```bash
uv run python tools/generate_staging_contract_fixtures.py
uv run python tools/generate_staging_contract_fixtures.py --check
```

## Repository provisioning

An authorized Hugging Face organization administrator provisions or reconciles
the repository with:

```bash
uv run python tools/provision_uk_staging_repository.py --apply
uv run python tools/provision_uk_staging_repository.py --verify-access
uv run python tools/provision_uk_staging_repository.py --verify-access --verify-write
```

The procedure creates the dataset repository with private visibility first,
then applies the Hugging Face setting `gated="manual"`, which means each access
request requires individual approval. If the settings update fails, the
recovery operation selects private visibility again and never selects public
visibility. The write probe is optional, uses only
`verification/operator-write-probe.json`, and deletes that probe after a
successful check.

The procedure was run successfully on 2026-09-08 while authenticated as
`anth-volk`, a member of the `policyengine` organization. The resulting
`policyengine/populace-uk-staging` dataset reported `private: true` and
`gated: "manual"`; in concrete terms, the repository is not publicly readable
and individual access requests require explicit approval. Anonymous repository
inspection was refused, authenticated repository inspection succeeded, and the
operator write probe succeeded and was removed. This confirms that the
organization supports the required combination of private visibility and
individual manual approval.

The bootstrap credential was then replaced on 2026-09-08 by the local token
named `microcosm-uk-staging-local-writer`. Hugging Face reported its role as
`fineGrained`; authenticated repository inspection and the temporary write
probe both succeeded, and the probe was removed. The token was configured by
the operator for this dataset only; its secret value was not printed or written
to the worktree. This establishes the scoped local-writer check, but the
complete access-control task still requires a separate authenticated user that
has not been approved to demonstrate refusal. Calibration Diagnostics and the
optional GitHub repository access check each require their own fine-grained
read-only credential.

## Command modes and files

The two UK commands (`tools/build_uk_frs_spine.py`, a shim over
`uk_runtime.spine_build`, and `microcosm-build-uk` / `tools/build_uk_full.py`
in either release role, `national` or `dense`; `tools/build_uk_rowwise_candidate.py`
is a stub over the same driver) support these staging modes:

- Default: local version 3 files plus best-effort delivery to
  `policyengine/populace-uk-staging`.
- `--staging-local-only`: the same validated files without constructing a
  remote client.
- `--no-staging`: no telemetry files, with an opt-out delivery object (delivery
  contract version 2) written into build evidence.

Common options are `--staging-dir`, `--staging-repo-id`,
`--staging-run-id`, `--staging-candidate-id`,
`--staging-upload-interval-seconds`, and `--staging-read-back`. An empty
repository identifier is invalid in remote mode. Authenticated read-back is
valid only in remote mode: the emitter commits the run's pending files and
compares the Hub's copy of each with the bytes it committed. Once read-back
has passed or failed, its verdict owns the delivery's `last_error_code`: the
refresh hand-over that follows a passed read-back is best effort, and if the
emitter does not accept it the run keeps its verified delivery (the miss shows
as an upload attempt without a success), while the remote copy may still show
the delivery as it was before read-back.

A document's delivery counts are the emitter's as of that document's
hand-over: `upload_successes` counts files the emitter committed and
`upload_attempts` the files it attempted to commit plus any it did not
accept. A document never counts the commit that carries it. Three hand-overs
in a row that the emitter does not accept pause the run's hand-overs; the
final state is handed over once more.

Version 2 and 3 storage use the fixed repository prefix `runs/`. Individual runs
cannot select another prefix, so consumers can enumerate that subtree without
traversing the rest of the repository.

Each local or remote run contains only files below its own run directory:

```text
runs/<run_id>/run_manifest.json
runs/<run_id>/progress.json
runs/<run_id>/events.ndjson
runs/<run_id>/calibration_progress.json  # calibration runs only
```

Every JSON document and event declares its schema name and version (3 for
new runs; version 2 runs stay valid, but one run never mixes versions). Unknown
schema names or versions are incompatible data.

A run ends in one of three terminal statuses:

- `completed`: the build finished and its gates passed;
- `blocked` (version 3): the build reached a gate decision and the gates refused
  the candidate. The run carries `block` (`phase`; `blocking_failure_count`, at
  least 1; `blocking_gate_ids`, empty when the refusal named no gate, never a
  placeholder) and ends with a `blocked` event that also lists the gate statuses
  when the refusing report could be read. A dense build refused at the preflight
  gates is blocked at phase `preflight`. A block whose details the staging
  content policy refuses closes the run `failed` with `GATE_BLOCK_UNRECORDED`
  instead, so no run is left `running`;
- `failed`: the build raised, or returned without a recorded block. `failure`
  carries an `error_code` (`INTERRUPTED`, `TERMINATED`, `OUT_OF_MEMORY`,
  `GRAPH_NODE_FAILED`, `BUILD_FAILED`, `BUILD_REFUSED`, `RUNG_ABORTED`,
  `GATE_BLOCK_UNRECORDED`) and, in version 3, a `failure_class`.

A stopped build is a failed run, not a running one. Ctrl-C closes it with
`INTERRUPTED`; a SIGTERM (a supervisor, a budget stop, `kill`) closes it with
`TERMINATED`, and the command then exits with status 143. Both record a
`discarded` Logbook row. The first SIGTERM is handled once: a second one kills
the process at once, even while the graph store is still settling. SIGKILL and
out-of-memory kills cannot be caught; for those the hosted emitter's heartbeat
reports `unexpected_process_exit`, and no Logbook row is written.

The delivery summary (`staging_delivery`) keeps contract version 2: its shape
did not change, and publication and the release assemblers pin it. Only reviewed aggregate JSON
artifacts are permitted; population H5 files, NumPy archives, source survey
tables, row-level extracts, archives, credentials, and environment data are
rejected before remote storage is called.

## Staged datasets

Telemetry is not the dataset. The build command (`microcosm-build-uk`; dense
exact-count `--dataset-households` and national runs alike) also stages the bundle its
manifest vouches for, so a run can be inspected by the team without being
published. The two destinations share one run id:

```text
policyengine/populace-uk-staging   runs/<run_id>/...          telemetry (above)
policyengine/populace-uk-private   staged/<run_id>/...        the bundle
```

The bundle is every file `rowwise_candidate_manifest.json` registers under
`outputs` (the H5, `calibration_diagnostics.json`, the gate report, the CSVs
and the registry), the manifest as built, `staged_manifest.json` (inventory
with digests, summary fields and the telemetry run reference) and
`sha256sums.txt`. Nothing else in the run directory is eligible: logs, the
Logbook spool, checkpoints and evaluation trees stay local. The bundle is
verified from disk against the manifest's own digests, then handed to the
build's emitter (`stage_dataset_bundle`): every file by path, with its SHA-256,
its size and its role. An output's role is its `outputs` key as a role name
(`dataset` for the H5, `calibration_diagnostics`, `local_registry`, ...; digests
in a key are left out), and the manifest, `staged_manifest.json` and
`sha256sums.txt` are `build_manifest`, `staged_manifest` and `checksums`. The
emitter writes the bundle in one commit with the build's own credential, and
its receipt becomes the bundle's `staged_dataset` record, with the commit as
its revision. `releases/` and `latest.json` are never touched, no tag is
created, and the release contract is not consulted: a staged bundle is not a
release and cannot be loaded as one.

Modes follow the staging switch. The default uploads telemetry and the bundle;
`--staging-local-only` keeps both on disk (the sidecars are still written);
`--no-staging` disables both and records the opt-out; `--no-staged-dataset`
runs telemetry alone, with the bundle neither inventoried nor uploaded. The
repository is `--staged-dataset-repo-id` (environment
`POPULACE_UK_STAGED_DATASET_REPO_ID`). Because the upload closes a multi-hour
run, the command asks its emitter to check the repository before it reads the
spine (`check_write_access`), and refuses to start a remote dataset stage
without a running emitter, or without an ambient Hub credential (`HF_TOKEN`)
that can see the repository **and** write it: a read token, or a fine-grained
token scoped to another owner, is refused up front rather than by the Hub's
403 hours later (a fine-grained token needs `repo.write` on the repository or
on the `policyengine` organisation). With `HF_HUB_OFFLINE` set, as the Modal
runner sets it for every build, the emitter could never upload, so a remote
stage stays local instead: the bundle is verified, its sidecars are written,
and it is recorded as `local_only`, to be staged from the workstation once
the outputs are copied back. The
telemetry stays best-effort with no pre-flight. Forwarded epochs are thinned to
at most 2,400 rows per run, whatever `--epochs` says, so the telemetry files
stay under the contract's 5 MiB cap; a content refusal from the telemetry is
reported once and stops the forwarding without touching the solve.

The build waits for the emitter's first upload attempt after the hand-over,
for at most 30 minutes (`STAGED_DATASET_WAIT_SECONDS` in
`microcosm.build.staging_dataset`); a healthy upload of the largest bundle
recorded (about 2.4 GB) takes about 16 minutes at 20 Mbit/s. The emitter
retries a failed attempt from its durable queue, but the build does not wait
for the retries: it records the bundle as `failed` with the attempt's error
code, or with `UPLOAD_PENDING` when the first attempt had not finished after
30 minutes. The emitter keeps such a bundle queued, and it or the next emitter
started on the machine (the re-stage command's included) uploads it. Leave the
run directory unchanged until then, and re-stage it to record the outcome. A
`failed` record therefore means one of two things. With `UPLOAD_PENDING`,
`REPOSITORY_UNAVAILABLE`, `UPLOAD_FAILED`, `CREDENTIAL_MISSING` or
`PRESERVED_FILES_MISSING` (`RETRYING_ERROR_CODES`), the bundle may still
arrive, because the emitter keeps it queued; with any other code
(`REMOTE_DIFFERS`, `PRESERVED_BYTES_CHANGED`, `HANDOVER_REFUSED`,
`EMITTER_UNAVAILABLE`, or `RECEIPT_UNRECOGNIZED` for a settled answer
Microcosm does not recognize) the failure is final. The manifest also names
the build's hand-over in `staged_dataset_handover`: `handover_id` (the
emitter's hand-over ID), `repository`, `run_id`, `run_dir` (the directory
whose paths the emitter's job names) and `files` (the outputs' digests and
sizes as handed over), so a re-stage can ask the emitter for its outcome. It
keeps what the build recorded in its staging run and Logbook row in
`staged_dataset_at_build` (`mode`, `status`, `error_code`), which re-stages
never change.

The manifest gains two evidence blocks after the bundle is on disk:
`staging_delivery`, the validated version 2 telemetry receipt, and
`staged_dataset` (`mode`, `repository`, `prefix`, `run_id`, `revision`,
`status` in `uploaded`, `already_staged`, `failed`, `skipped`, a reviewed
`error_code`, and the `files` inventory as a mapping). The error codes are the
emitter's (`REMOTE_DIFFERS` and `PRESERVED_BYTES_CHANGED`, which are final;
`REPOSITORY_UNAVAILABLE`, `UPLOAD_FAILED`, `CREDENTIAL_MISSING` and
`PRESERVED_FILES_MISSING` for a bundle it keeps retrying) and four of
Microcosm's: `UPLOAD_PENDING`, `HANDOVER_REFUSED` for a hand-over the emitter
refused to queue, `EMITTER_UNAVAILABLE` when no emitter took it, and
`RECEIPT_UNRECOGNIZED` for a settled answer it does not recognize. A failed
upload is recorded with its code, warned on stderr, and never changes the
build's exit code or Logbook disposition. The emitter copies the bundle's small
files (each up to 32 MiB, up to 64 MiB in all, smallest first) before it
answers, and uploads every other file from disk. So after staging the build
rewrites the manifest, `sha256sums.txt` and the national build record only when
the receipt lists them in `copied_files`, as it does for today's bundles; a
file it does not list keeps its handed-over bytes, with a warning, and the
records it would have gained are in the staging run's artifacts and in the
manifest the command prints. The telemetry run declares two reviewed
artifacts, `artifacts/staged_dataset.json` (the same block) and
`artifacts/fit_summary.json` (loss, fit by family, gate verdicts, the size
receipt without its per-row arrays), so a run in the dashboard points at its
bundle. A national run given `--incumbent-h5` adds a third,
`artifacts/score_vs_incumbent.json`: the rule-1 score receipt against the
incumbent on the surface both can materialize, produced after the bundle is
staged and before the telemetry completes, with the rows the incumbent cannot
materialize pruned from both arms and listed; the manifest's `evaluation`
block records its verdict, and the release-cut certifier requires that
verdict to be `passed`.

Re-stage a finished directory, including runs built before this lane existed,
runs that kept their bundle local, and runs whose upload failed or was still
queued, with:

```bash
uv run python tools/stage_uk_rowwise_candidate.py --run-dir <run directory>
```

It starts a short-lived emitter of its own, with only its Hugging Face module,
and checks write access as a build does. It then works out what to do from the
manifest:

- **The record says the bundle landed.** A manifest that records the same
  outputs as uploaded, or the outputs the build handed over while the
  directory still holds them (a national build's build record and score
  receipt may differ, since it writes them after the hand-over), is left
  untouched: the driver's revision stands. A landed record is never replaced
  by a failure or a local-only record; when an attempt failed (the directory
  changed since, or the stage went to another repository) the record stays
  and the command exits 1.
- **The emitter still holds the build's hand-over.** The manifest's
  `staged_dataset_handover` names it. Emitters run by the same user with the
  same cache
  directory share their hand-overs: the hand-over database,
  `events.handovers.sqlite3`, beside the emitter's event queue file under
  `$XDG_CACHE_HOME/microcosm/telemetry/` (`~/.cache` when `XDG_CACHE_HOME` is
  unset). The command reads that hand-over's receipt first. One that landed is
  recorded as `already_staged`, with its commit, wherever the directory is
  now. One still queued is waited for; one the emitter settled as a failure
  is recorded with its final code (`REMOTE_DIFFERS`, for example when a
  re-stage of a copy of the directory uploaded first). Nothing is handed over
  again. That matters for a national build, which rewrites its build record
  after the hand-over: a bundle built again from the directory would differ
  from the one handed over.
- **That hand-over has not landed and can never finish.** Its job reads the
  files the emitter did not copy from the paths they had at hand-over, in
  `run_dir`. If one of them is gone from there (the directory moved, rather
  than being copied) or the emitter reports the files changed
  (`PRESERVED_BYTES_CHANGED`), the command hands the directory, where it is
  now, over as a new job, and records that hand-over. A job that reported
  files missing (`PRESERVED_FILES_MISSING`) while they are back at its paths
  is waited for instead. The emitter keeps retrying an abandoned job for as
  long as emitters run on the machine; the wait between attempts doubles from
  2 seconds until it reaches its cap of five minutes. No emitter command
  removes a job, and Microcosm cannot cancel it.
- **Otherwise** (another machine, another user or cache directory, a cleared
  cache, or a run that kept its bundle local) it hands the directory over. For
  a national build whose bundle already reached the Hub, that hand-over comes
  back `REMOTE_DIFFERS`: the folder holds the bundle with the build record as
  handed over, and this directory's build record has changed since. The
  emitter cannot say whether the folder holds this build's own bundle, so the
  command keeps the manifest's records, exits 1 and says so. Re-stage where
  the build's emitter keeps its hand-overs to record it, or compare
  `staged/<run_id>/staged_manifest.json` on the Hub with the hand-over record's
  `files`. Do not stage it under another run id, which would upload a second
  copy.

It waits for the emitter's upload attempt for up to three hours by default
(`--wait-seconds`; a 2.4 GB bundle takes about 2 hours 40 minutes at
2 Mbit/s), longer than a build's 30 minutes, since finishing the upload is its
only job. When it stops waiting it closes its emitter, which abandons an upload
still running; the next emitter started on the machine retries it from the
start. The command updates the manifest's record. It cannot update the build's
other records of a failure, and says so: the staging run's `staged_dataset`
and `fit_summary` artifacts and its `dataset_staging` event still hold what
the build recorded, from `staged_dataset_at_build` (a completed staging run
accepts no changes), and the Logbook row still lists the phase that status
gave it, `dataset_stage_failed` or, for a build that kept its bundle local,
`dataset_stage_skipped` (a row is identified by the digest of its content and
chained to the next row). So the dashboard keeps
showing such a run without its bundle. An identical remote bundle the manifest
does not know about is recorded as `already_staged` with the bundle's own
commit; a different bundle
under the same run id is refused (`REMOTE_DIFFERS`).

No UK program is registered with the Modal runner yet; the runner's
programs are all US builds today (`tools/modal_us_stage_plan.py`), and none of
them writes a staged-dataset bundle. A UK build run there would stage nothing
from its container: the runner sets `HF_HUB_OFFLINE=1` for every tool, so the
build would record its bundle as `local_only` (see above). After `modal volume
get` has copied such a run back
([the Modal runbook](us-modal-stage-runbook.md#commands), step 5), you would
stage it from the workstation, with your own emitter and Hub credential, by
naming the stage's receipt:

```bash
uv run python tools/stage_uk_rowwise_candidate.py \
  --run-dir ./modal-runs/<run_id>/state/<output directory> \
  --modal-receipt ./modal-runs/<run_id>/receipts/<stage>-<utc>.json
```

Before it reads any output, the command checks that the receipt is of a
stage that finished (`COMPLETED`, and not stopped at its budget) and that the
run directory, wherever a link or `..` in its path leads, is inside the run's
state directory: `state/` beside the receipt's `receipts/` directory, or the
directory `--modal-state-root` names, either resolved the same way. The
receipt must list the build manifest and every output it registers, at their
paths under the state directory. An output the stage hashed but never copied
to the runs volume (the receipt's `outputs_not_mirrored`) is named as such.
Unless the build manifest's `staged_dataset` record is `local_and_remote`,
the manifest must have the receipt's SHA-256 and size, so a manifest edited to
register another file the stage wrote, rename a role or change any other field
is refused. A hand-over from the workstation writes a `local_and_remote`
record and rewrites the manifest, so after one, even one that failed, the
manifest is not compared. The command reads that mode from the manifest
itself, so it cannot tell a record a hand-over wrote from one edited to say
`local_and_remote`: such an edit also ends the manifest's comparison, and the
edits above are then not caught. Every output must have the receipt's SHA-256
and size whatever the record says, so an output whose bytes differ from the
receipt's is always refused. Every refusal comes before an emitter starts.
`staged_manifest.json` and `sha256sums.txt` are not compared, because staging
writes them again. Then the command stages the directory as above, under the
build's own run id: the hand-over, the records, a re-stage after a failed
attempt and a second run on a landed bundle work as they do for a directory
built on the workstation. Unset `HF_HUB_OFFLINE` first if your shell sets it:
the command refuses to stage while it is set. With `--local-only` it checks
the directory against the receipt and writes `staged_manifest.json` and
`sha256sums.txt`, but leaves the build manifest as it is, so later runs with
the receipt still compare it. (Without `--modal-receipt`, `--local-only` also
records the bundle as `local_only` in the build manifest.) Nothing here
submits the bundle to the collector: the
command's emitter runs its Hugging Face module only, and builds do not submit
artifacts to the collector's artifact intake yet
(PolicyEngine/microcosm-collector#12).

Fetch a bundle for a scorecard, an evaluation leg or the
dashboard's local-directory mode with:

```bash
uv run python tools/fetch_uk_staged_dataset.py --run-id <run_id> --dest <dir>
```

Every file is checked against `sha256sums.txt`; `--h5-only` fetches the
dataset alone. Rollback is configuration-first here too: `--no-staged-dataset`
or `--staging-local-only` for new runs, and deleting `staged/<run_id>/` on the
Hub for a bundle that must not remain (its manifest keeps the record). The
dense release assembler applies the national rule to this command's runs: it
requires the manifest's `staging_delivery` receipt and copies it into
`build_manifest.json` as `staging`, where publication reads it. A dense run
built before this lane carries no receipt; `--allow-missing-staging` assembles
it with a recorded disabled-staging opt-out naming the override, the same
posture publication's `--allow-missing-staging` grants. Since microcosm#823 the
pre-flight and the assembler check the manifest's `release_role` before that
override, so a run built before the release role existed is refused on the
role and is rebuilt, never grandfathered.

## Smoke verification

The UK spine command (`tools/build_uk_frs_spine.py`, a shim over the package's
`uk_runtime.spine_build`) keeps fractional input sampling for scale tests. The
`--smoke` option marks its H5, sidecar, and staging records as non-release. It
does not invoke national calibration, release certification, release assembly,
or publication.

Continuous integration runs all current UK spine transformations against the
complete deterministic synthetic fixture. It does not reduce the fixture by a
source-family count. Every output stays below the runner's temporary directory,
and the workflow receives no external writer credential.

Run the integration command locally with:

```bash
uv run python tools/build_uk_frs_spine.py \
  --synthetic-fixture-dir packages/microcosm-graph/tests/fixtures/parity/uk_spine/sources \
  --spine-h5 <temporary-directory>/uk-smoke.h5 \
  --sample-fraction 1.0 \
  --sample-seed 42 \
  --smoke \
  --staging-local-only \
  --staging-dir <temporary-directory>/staging \
  --staging-run-id ci-uk-smoke-full-s42
```

The `integration-uk` job in `.github/workflows/test.yml` runs when unified
changed-path selection includes the UK and on every push to `main`. Its result
is reported directly as a workflow check. Its commands live in
`tools/run_integration_tests.sh`.
The test reports total elapsed time and the elapsed time for each
transformation.

An authorized operator can verify the current remote layout with the same
synthetic fixture by omitting `--staging-local-only`, adding
`--staging-read-back`, and supplying unique `--staging-run-id` and
`--staging-candidate-id` values. The command still requires `--smoke`, so its
H5 file, sidecar, and aggregate staging records are marked non-release. The H5
file and fixture source tables remain local because the staging content policy
permits only the reviewed aggregate JSON files.

An authenticated staging transport check completed on 2026-09-09 using the
earlier source-family-count interface and the superseded shared-file layout. It
uploaded version 2 JSON records for `uk-smoke-h0100-s578-20260909T125305Z` to
`policyengine/populace-uk-staging`; it did not upload the H5 dataset. Its
authenticated read-back does not verify the current run-scoped-only layout.
That obsolete sampling option has also been removed because it selected source
families before construction and therefore did not guarantee a requested final
household count.

A current-layout authenticated check completed on 2026-09-14 using the full
deterministic synthetic fixture. It recorded
`uk-smoke-full-s42-20260914T172919Z` under its own `runs/` directory, completed
15 of 15 upload attempts, and passed authenticated read-back. Repository
inspection found exactly `run_manifest.json`, `progress.json`, and
`events.ndjson` in that directory. The Calibration Diagnostics PR preview then
listed and loaded the completed non-release run through its server API while
reporting the obsolete 2026-09-09 run separately as incompatible.

## Monitoring authentication

Calibration Diagnostics already loads staging files inside Next.js API routes,
and its Hugging Face authorization header is constructed only in server-side
library code. UK support will retain that request boundary and use separate
server deployment variables:

- `POPULACE_UK_STAGING_HF_REPO`
- `POPULACE_UK_STAGING_HF_REVISION`
- `POPULACE_UK_STAGING_HF_TOKEN`

The browser calls the Calibration Diagnostics API routes and never calls the
private Hugging Face repository with a credential. Responses and error text
must not contain the token. The UK token is a fine-grained read credential for
`policyengine/populace-uk-staging`; it is not shared with the local writer.

The version-aware consumer implementation was published for review on
2026-09-08 as
[`PolicyEngine/calibration-diagnostics#181`](https://github.com/PolicyEngine/calibration-diagnostics/pull/181).
Its feature-branch checks passed locally with TypeScript compilation and all
328 tests before publication; its GitHub tests, TypeScript/build check, and
Vercel preview deployment also passed. Review was requested from `@MaxGhenis`.
UK remote output remains disabled until that change is reviewed, merged,
deployed with the separate read credential, and verified against both contract
fixture versions.

## Access-control verification

Hugging Face currently documents both `private=True` and `gated="manual"` on
the repository-settings API, and documents individual approval for gated
datasets. This establishes API support, but not the effective behavior of the
PolicyEngine organization. Before remote output is enabled, the provisioning
procedure must verify the combined settings with an organization writer and
then prove anonymous or unapproved refusal, approved authenticated access, and
narrowly scoped writer access. If any check fails, the repository remains
private and UK remote output remains disabled.

References:

- <https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.update_repo_settings>
- <https://huggingface.co/docs/hub/datasets-gated>
- <https://huggingface.co/docs/hub/repositories-settings>

## Monitoring, rollback, and publication

Consumers discover runs by enumerating `runs/*/run_manifest.json`, then derive
the latest run from manifest `updated_at` timestamps. Monitor each run's
`progress.json` for current status and `events.ndjson` for ordered stage
durations. A remote verification run downloads and validates only its own run
manifest and progress document. Hand-over and commit failures are recorded
with counts and reviewed error codes while local recording continues;
unrestricted remote exception text is not serialized.

Rollback is configuration-first: use `--staging-local-only` to retain local
evidence or `--no-staging` for a deliberate opt-out, disable the GitHub
workflow, and revoke its optional read credential. Revoke the local writer
credential separately. Leave existing private run evidence intact for audit
and do not alter production repository references.

National calibration build records carry the validated version 2 delivery
object. Release assembly copies that object unchanged into
`build_manifest.json`. Normal publication refuses local-only evidence, invalid
or unknown delivery versions, and remote mode with zero successful uploads. It
retains the existing explicit missing-staging override and accepts a valid
deliberate opt-out. Production publication remains a separate operator action.
