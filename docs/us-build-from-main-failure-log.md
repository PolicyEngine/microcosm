# Failure log for a US build from main

Agents record here each gate that stops a US national build from `main`, in the order a run reaches it, so the next
attempt starts with every known stop in view. Each entry names what refuses, links its lines at the commit an agent
read, gives the message or the condition, and carries one status in its heading. `from the code` marks a refusal read in
the code and not run. `predicted` marks a stop that the code, together with a run's settings or its host, points to.
`observed` marks a stop a run hit, and the entry then names that run's commit and its log line.

## The run

Every entry cites lines at commit
[c7a5f96796ac6a16e7c520925e2517e9abbc8512](https://github.com/PolicyEngine/microcosm/commit/c7a5f96796ac6a16e7c520925e2517e9abbc8512).
A run starts with `ROUTE_A_ENV=<settings file> bash tools/route_a/route_a.sh` from a checkout at that commit
([route_a.sh](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L1-L13)).
The driver resolves its inputs and checks the commit, creates a worktree and syncs it, and prefetches the donors.
It then builds the base, or takes a Modal base that the hand-off accepted, and preflights it. Last, it builds the
release on the dense `--base-h5` arm, preflights the release and runs the publisher's offline preflight, and it
publishes nothing. An agent who runs a later commit rechecks each cited line there first.

[#1650](https://github.com/PolicyEngine/microcosm/pull/1650) merged after that commit.
On `main`, the driver now passes `--python` to the sync and unsets `UV_FROZEN` inside it
([route_a.sh#L384-L387](https://github.com/PolicyEngine/microcosm/blob/17fecf5e3434f649a20bfa09814b120593bbcaa1/tools/route_a/route_a.sh#L384-L387)),
and it checks the installed emitter against the commit's minimum when it checks the commit
([route_a.sh#L248-L269](https://github.com/PolicyEngine/microcosm/blob/17fecf5e3434f649a20bfa09814b120593bbcaa1/tools/route_a/route_a.sh#L248-L269)).
It also exports `MICROCOSM_TELEMETRY_REQUIRED` when the settings file sets it
([route_a.sh#L51-L54](https://github.com/PolicyEngine/microcosm/blob/17fecf5e3434f649a20bfa09814b120593bbcaa1/tools/route_a/route_a.sh#L51-L54)),
and its feed and crosswalk blockers name the commit's own pins
([route_a.sh#L176-L199](https://github.com/PolicyEngine/microcosm/blob/17fecf5e3434f649a20bfa09814b120593bbcaa1/tools/route_a/route_a.sh#L176-L199)).

[#1653](https://github.com/PolicyEngine/microcosm/pull/1653) merged after that commit too. On `main`, `sha_of` fails
when `openssl` prints no digest
([route_a.sh#L124-L132](https://github.com/PolicyEngine/microcosm/blob/a21fa3b9be33c0e24601989f041a9fdd41de704e/tools/route_a/route_a.sh#L124-L132)),
and `verify_inputs` then records `$role could not be read to hash it: $path`
([route_a.sh#L150-L151](https://github.com/PolicyEngine/microcosm/blob/a21fa3b9be33c0e24601989f041a9fdd41de704e/tools/route_a/route_a.sh#L150-L151)).

## Failures in run order

### 1. `verify_inputs` finds a processed input missing from `STORAGE`, or gets no digest for an input (`predicted`)

The driver hashes six processed inputs under `STORAGE` on every run, a Modal base included: the three ASEC years,
the PUF pair and ACS 2022
([route_a.sh#L92-L97](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L92-L97)).
When one of them sits outside that directory, `verify_inputs` records `$role missing: $path`
([route_a.sh#L127-L128](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L127-L128)),
and the run stops with `input problems (see above)`
([route_a.sh#L326-L327](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L326-L327)).
The driver checks the inputs again after it waits for the overnight chain
([route_a.sh#L388-L389](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L388-L389)),
and `--resolve` exits 2 on the same problems.

`sha_of` discards the hashing program's error output
([route_a.sh#L116-L116](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L116-L116)),
so an input that program cannot open reaches `verify_inputs` as an empty digest and a `sha256` mismatch
([route_a.sh#L134-L136](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L134-L136)).
On 17 November 2026 (UTC), in macOS launchd jobs, `/usr/bin/shasum`, `/usr/bin/openssl` and `/usr/bin/python3` could
not open files on an external drive (`Operation not permitted`). `sha_of` runs the first `openssl` on `PATH`. In
those jobs a Homebrew build came first and read every input on that drive, so `--resolve` reported no input
problems. The worktree's Python read a file there that `/usr/bin/python3` could not open.

### 2. A rerun from `main` must set the feed and crosswalk pins, or the driver refuses (`from the code`)

The driver keeps the published run's feed and crosswalk pins as its defaults on purpose
([route_a.sh#L55-L56](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L55-L56),
[#L77-L78](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L77-L78)).
Its comment says a rerun's settings must pin both at the chosen commit when `main` has changed them
([route_a.sh#L53-L54](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L53-L54)).
The settings template tells a rerun to set `FEED_SHA`, `FEED_BYTES` and `CROSSWALK_SHA` beside `FEED`, after
checking that commit's packaged pins
([route_a.env.example#L19-L21](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.env.example#L19-L21)).
The driver's test pins that choice, with the published digests when the settings leave them unset and the settings'
values when they set them
([test_route_a_driver.py#L138-L174](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/tests/engine_free/us/test_route_a_driver.py#L138-L174)).

At the pinned commit, `us/chronicle_feed.json` pins another `facts_sha256`
([chronicle_feed.json#L14-L14](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json#L14-L14)),
and the packaged crosswalk hashes to another digest, so the driver refuses a run from that commit that keeps the
defaults. With the template's `FEED` and the default pins, `check_commit` records
`us/chronicle_feed.json at $full does not pin facts_sha256 $FEED_SHA (the release tool would refuse the feed)` and
`packaged CD vintage crosswalk at $full is not $CROSSWALK_SHA`
([route_a.sh#L168-L182](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L168-L182)).
A settings file that points `FEED` at main's feed but keeps the default digest and size also fails `verify_inputs`
on that feed's size and digest
([route_a.sh#L130-L137](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L130-L137)).

The driver runs `verify_inputs` and then `check_commit`, and it logs every problem before it stops
([route_a.sh#L322-L327](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L322-L327)).
A run exits on `input problems (see above)` first and on `commit blockers (see above)` next, while `--resolve` exits
2 on input problems and only logs commit blockers
([route_a.sh#L293-L298](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L293-L298)).
The 18 October rerun runbook's first script stops sooner, with
`REFUSED: main changed the release feed; review its pins/settings`
([us-route-a-rerun-20261018.md#L191-L192](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/docs/us-route-a-rerun-20261018.md#L191-L192)).

### 3. `uv sync` stops on the host's interpreter or on `UV_FROZEN` (`predicted`)

`ensure_worktree` runs `"$UV" sync --all-packages --locked --extra us` with no `--python`, and it stops with
`uv sync in $WT` when the sync fails
([route_a.sh#L344-L346](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L344-L346)).
The pinned commit has no `.python-version`, and `pyproject.toml` sets `requires-python = ">=3.13"` with no upper
bound
([pyproject.toml#L5-L5](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/pyproject.toml#L5-L5)).
The locked torch 2.12.0 carries wheels for CPython 3.13 and 3.14 and no source archive
([uv.lock#L2319-L2357](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/uv.lock#L2319-L2357)),
so the sync fails when uv picks a CPython 3.15 interpreter on the host. The driver sources the settings file
([route_a.sh#L32-L34](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L32-L34)),
so `export UV_PYTHON=3.14` there pins the interpreter.

The same sync stops in a shell that exports `UV_FROZEN=1`, since uv refuses `--locked` beside it. Version 0.11.7 exits
2 while it parses its arguments and prints this error:

```
error: the argument `--locked` cannot be used with `UV_FROZEN` (environment variable)
```

When an agent starts the driver from such a shell, it stops at the same `uv sync in $WT`. On 17 November 2026 (UTC)
an agent ran the same sync command outside the driver at the pinned commit, with `UV_PYTHON=3.14` and without
`UV_FROZEN`, and it completed.

### 4. The base needs 80 GiB of available RAM locally, or a lineage match on Modal (`predicted`)

The driver admits a local base only with 80 GiB of available RAM and, before any checkpoint exists, 70 GiB of
free disk, and it caps the base at 100 GiB RSS
([route_a.sh#L518-L535](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L518-L535)).
`wait_resources` waits up to `RESOURCE_WAIT_HOURS`, 12 by default, and then fails with
`needs 80 GiB available RAM and <n> GiB free disk`
([route_a.sh#L47-L47](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L47-L47),
[#L393-L400](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L393-L400)).
The driver's comment records 44 GB of frame checkpoints from a base measured on 16 September
([route_a.sh#L511-L514](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L511-L514)),
and it removes them after hashing the base only with `PRUNE_BASE_CHECKPOINTS=1`
([route_a.sh#L542-L545](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L542-L545)).

The driver skips the base stage when the run folder holds `base-sup/ACCEPTED`
([route_a.sh#L432-L432](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L432-L432),
[#L518-L518](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L518-L518)),
which the Modal hand-off writes last
([README.md#L178-L187](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/README.md#L178-L187)).
The hand-off's `compare-lineage` requires the Modal run to match a local reference, source digest included
([us-modal-stage-runbook.md#L1230-L1254](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/docs/us-modal-stage-runbook.md#L1230-L1254)).
The committed reference records a local run at an older commit
([reference#L6-L6](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/docs/us-modal-stage-route-a-base-local-reference.json#L6-L6),
[#L216-L216](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/docs/us-modal-stage-route-a-base-local-reference.json#L216-L216);
[README.md#L189-L194](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/README.md#L189-L194)),
so a Modal base at the pinned commit needs a local reference made at that commit first. When the stage-001
checkpoints differ, `lineage_class_check.py` exits 1 on any differing column other than `person.pre_subsidy_rent`,
even inside its 1e-12 relative tolerance
([lineage_class_check.py#L112-L122](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/lineage_class_check.py#L112-L122),
[#L138-L138](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/lineage_class_check.py#L138-L138)).

### 5. The installed emitter predates the 0.16.0 minimum (`predicted`)

A build starts the separately installed `microcosm-emitter` as its child process, for telemetry, Orrery graph
publication and the Hugging Face hand-over of staging run files
([telemetry_emitter.py#L1-L8](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L1-L8)).
The adapter never starts an emitter older than the minimum that `minimum_emitter_version.json` sets
([telemetry_emitter.py#L35-L38](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L35-L38),
[minimum_emitter_version.json#L2-L2](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/minimum_emitter_version.json#L2-L2)).

With `MICROCOSM_TELEMETRY_REQUIRED=1`, an emitter the adapter cannot use stops the build before it does any work
([telemetry_emitter.py#L24-L28](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L24-L28),
[#L386-L401](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L386-L401)).
The module and the README both say PolicyEngine's own builds keep telemetry required
([telemetry_emitter.py#L31-L33](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L31-L33),
[README.md#L137-L141](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/README.md#L137-L141)).
For an older emitter, the adapter raises `TelemetryRequiredError` with reason `outdated_emitter`, saying the
installed emitter
`reports version {version}, and this build needs {MINIMUM_EMITTER_VERSION} or later`
([telemetry_emitter.py#L444-L461](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L444-L461),
[#L572-L573](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L572-L573)).
A local base starts the first emitter, after parsing its arguments and before reading any input
([build_us_puf_support_base.py#L8-L12](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/build_us_puf_support_base.py#L8-L12),
[#L865-L870](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/build_us_puf_support_base.py#L865-L870)).
After a Modal base, the release starts the first emitter once it has parsed its arguments
([build_us_fiscal_refresh_release.py#L13434-L13446](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/build_us_fiscal_refresh_release.py#L13434-L13446)),
so on that path the base preflight of entry 6 runs first
([route_a.sh#L560-L566](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L560-L566)).

The driver never sets the variable itself, since it gives its base and release stages only `PYTHONUNBUFFERED`
([route_a.sh#L478-L479](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L478-L479)).
The supervisor passes the caller's environment through
([supervise.py#L84-L85](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/supervise.py#L84-L85)).
The driver loads the settings file without exporting its values
([route_a.sh#L32-L34](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L32-L34)),
so a stage sees the requirement only when the run's environment or an `export` line carries it.
Unset or `0`, a missing, refusing or outdated emitter leaves the build running without it
([telemetry_emitter.py#L14-L15](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L14-L15)).
For an older emitter the adapter prints a warning, and the build continues without hosted telemetry or staging
hand-overs and queues no Orrery graph publication
([#L192-L202](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L192-L202),
[#L574-L583](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L574-L583)).

### 6. The preflight can stop the driver on exit 1 (`from the code`)

The driver runs `tools/preflight_us_release_gates.py` on the base with `--ledger-facts "$FEED"`, and it accepts exit 0
(clean) or 2 (at risk) unless `RELEASE_DESPITE_PREFLIGHT_FAIL=1`
([route_a.sh#L550-L566](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L550-L566)).
Given `--ledger-facts`, the preflight compiles the fiscal-target registry the release calibrates, with age targets
and the packaged crosswalk, and previews zero support on it beside its export-mass, smoke-support and SPM
composition checks
([preflight_us_release_gates.py#L308-L315](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/preflight_us_release_gates.py#L308-L315),
[#L373-L383](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/preflight_us_release_gates.py#L373-L383),
[release_gate_preflight.py#L943-L977](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/us_runtime/release_gate_preflight.py#L943-L977),
[#L1063-L1074](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/us_runtime/release_gate_preflight.py#L1063-L1074)).
The feed's re-pin record lists one compile of main's feed, 33,250 targets with 6,102 on the `national_state` surface
([us-chronicle-feed-repin.md#L170-L186](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/docs/us-chronicle-feed-repin.md#L170-L186)).

On 17 November 2026 (UTC) an agent ran the preflight outside the driver, offline, at the pinned commit, on an
existing base built at 9434c504b, with `--new-lineage`, and it exited 2, which the driver accepts. It skipped
selection carry-over and passed its capital-gains tail, smoke-support and SPM composition checks. Its zero-support
preview could statically check none of the 33,193 positive targets, so it passed without grading any. Its
export-mass check found two columns under their bands at base weights.
`partnership_income` held $394.2 billion against a band of $423.9 billion to $1,271.6 billion, and
`social_security_survivors` held $67.8 billion against $80.5 billion to $241.5 billion. The preflight asks an agent
to confirm that calibration targets each column; this run measured neither column after calibration.

### 7. The QRF tail register must match the new export's concentrated columns (`predicted`)

At its terminal batch, after calibration, the release evaluates the QRF tail-concentration gate on the export and
raises `Release gates failed: ...` on any failure in the batch
([build_us_fiscal_refresh_release.py#L16757-L16819](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/build_us_fiscal_refresh_release.py#L16757-L16819)).
A sparse QRF-imputed column whose top 100 weighted records carry over 75% of its mass fails unless the register
lists it, and a listed column at or under that share fails as stale
([gates.py#L3436-L3457](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/gates.py#L3436-L3457)).
A listed column that the gate found stale or never checked adds
`QRF tail-concentration register mismatch: the per-run exclusion register must exactly match the concentrated columns`
to the batch, whatever `--allow-qrf-tail-concentration` says
([build_us_fiscal_refresh_release.py#L8557-L8590](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/build_us_fiscal_refresh_release.py#L8557-L8590),
[#L8619-L8630](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/build_us_fiscal_refresh_release.py#L8619-L8630)),
and the driver refuses any `--allow-*` flag in `RELEASE_EXTRA_ARGS`
([route_a.sh#L241-L245](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L241-L245)).
The template's default register comes from the published run
([route_a.env.example#L47-L48](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.env.example#L47-L48),
[README.md#L62-L64](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/README.md#L62-L64)),
and the driver's comment asks for a register measured on the export that ships
([route_a.sh#L68-L71](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L68-L71)).
A base built at the pinned commit brings its own concentrated columns, so a register from another export
will likely fail here.

### 8. If #1635 merges first, its gate shadow can fail a finished base or release (`predicted`)

[#1635](https://github.com/PolicyEngine/microcosm/pull/1635), unmerged at the pinned commit, runs each US gate's
`us/gates.json` entry in shadow beside today's gate. When a tool finishes, the shadow fails the run on a verdict
difference that no threshold override explains
([gate_shadow.py#L11-L16](https://github.com/PolicyEngine/microcosm/blob/2c34c57f0628a97598dda0c020242d429d6bb01b/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L11-L16),
[#L33-L39](https://github.com/PolicyEngine/microcosm/blob/2c34c57f0628a97598dda0c020242d429d6bb01b/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L33-L39),
read at the pull request's
[head commit](https://github.com/PolicyEngine/microcosm/commit/2c34c57f0628a97598dda0c020242d429d6bb01b)).
Every real run arms it unless `MICROCOSM_US_GATE_SHADOW=off`
([#L41-L44](https://github.com/PolicyEngine/microcosm/blob/2c34c57f0628a97598dda0c020242d429d6bb01b/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L41-L44),
[#L114-L124](https://github.com/PolicyEngine/microcosm/blob/2c34c57f0628a97598dda0c020242d429d6bb01b/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L114-L124)),
and #1635 arms it in both the base tool
([build_us_puf_support_base.py#L992-L994](https://github.com/PolicyEngine/microcosm/blob/2c34c57f0628a97598dda0c020242d429d6bb01b/tools/build_us_puf_support_base.py#L992-L994))
and the release tool
([build_us_fiscal_refresh_release.py#L13820-L13820](https://github.com/PolicyEngine/microcosm/blob/2c34c57f0628a97598dda0c020242d429d6bb01b/tools/build_us_fiscal_refresh_release.py#L13820-L13820),
[#L13438-L13445](https://github.com/PolicyEngine/microcosm/blob/2c34c57f0628a97598dda0c020242d429d6bb01b/tools/build_us_fiscal_refresh_release.py#L13438-L13445)).
The driver accepts only exit 0 from the base and from the release
([route_a.sh#L534-L534](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L534-L534),
[#L585-L587](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L585-L587)).
