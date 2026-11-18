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

### 2. The driver needs the original SCF 2022 summary extract, which the Federal Reserve replaced (`predicted`)

`scf_wealth.py` pins two digests for the SCF 2022 summary extract, one for the zip at
`https://www.federalreserve.gov/econres/files/scfp2022s.zip` and one for its member `rscfp2022.dta`
([scf_wealth.py#L161-L176](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py#L161-L176)).
The Federal Reserve replaced that zip on 9 October 2026
([#1368](https://github.com/PolicyEngine/microcosm/issues/1368)). On 18 November 2026 (UTC), `curl -sI` of the URL
reported a length of 3,715,841 bytes and a last-modified date of 9 October 2026. The same day, a
[comment on #1368](https://github.com/PolicyEngine/microcosm/issues/1368#issuecomment-6966217460) reported the
sha256 of the replacement's member and its size, 34,487,960 bytes.

`verify_inputs` checks the file that `SCF` names against the member pin and the original member's size, 24,904,185
bytes
([route_a.sh#L106-L106](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L106-L106)).
It records `$role missing: $path` for a missing file, and `$role size $asize != $size: $path` and
`$role sha256 $actual != $sha: $path` for the replacement zip's member
([route_a.sh#L127-L137](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L127-L137)).
The run then stops with `input problems (see above)`, as in entry 1.

The prefetch stage calls `fetch_scf_2022_summary_extract()` with no arguments, so the fetch ignores `SCF` and looks
in its own cache, `~/.cache/microcosm/scf`
([route_a.sh#L488-L502](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L488-L502),
[scf_wealth.py#L521-L527](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py#L521-L527)).
Without a network call, it returns a cached `rscfp2022.dta` whose digest equals the member pin
([#L528-L533](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py#L528-L533)).
Otherwise it downloads the zip and checks the zip pin before the member pin
([#L535-L558](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py#L535-L558)).
The replacement zip, whose digest #1368 gives as `d4e139c6…`, fails that check, and the fetch raises
`SCF 2022 summary-extract zip failed sha-256 verification: expected {expected_zip_sha256}, got {digest}.`
([#L541-L547](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py#L541-L547)).
The driver logs a first failure and goes on, with
`prefetch failed (see $RUN/prefetch-sup/run.log); the base proceeds and the release retries the prefetch`
([route_a.sh#L503-L508](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L503-L508)).
After the base preflight of entry 7, the driver runs the prefetch again and stops on a second failure, before the
release starts
([#L575-L576](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L575-L576),
[#L470-L475](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L470-L475)).
It logs `FAIL: prefetch <result>; log $RUN/prefetch-sup/run.log`, with the start of the stage's `RESULT.json` as
`<result>`
([#L111-L115](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L111-L115)).
The supervisor writes the stage's output, the fetch's zip message included, to that `run.log`
([supervise.py#L96-L102](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/supervise.py#L96-L102)).

A driver run passes both checks when the file `SCF` names and the cached member match the member pin. The
release then reads `SCF` through `--scf-summary-extract` in place of the fetch
([route_a.sh#L277-L277](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L277-L277),
[build_us_fiscal_refresh_release.py#L14805-L14818](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/build_us_fiscal_refresh_release.py#L14805-L14818)).
A release-tool run without that flag calls the same fetch
([#L1741-L1752](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/build_us_fiscal_refresh_release.py#L1741-L1752)),
and on a machine whose cache lacks the original member it stops at the zip check. `--resolve` runs `verify_inputs`
but never the prefetch
([route_a.sh#L287-L298](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L287-L298)),
so it passes a run whose cache lacks the original member. The fetch checks a cached member against the member pin
alone, so the prefetch passes with a copy of the original member at `~/.cache/microcosm/scf/rscfp2022.dta`. On a
machine with no copy, the driver has no setting that passes either check, and #1368 tracks pointing the fetch at a
copy of the original zip.

### 3. A rerun from `main` must set the feed and crosswalk pins, or the driver refuses (`from the code`)

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

### 4. `uv sync` stops on the host's interpreter or on `UV_FROZEN` (`predicted`)

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

### 5. The base needs 80 GiB of available RAM locally, or a lineage match on Modal (`predicted`)

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

### 6. An emitter older than the 0.16.0 minimum stops a build that requires telemetry (`from the code`)

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
so on that path the base preflight of entry 7 runs first
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

On 18 November 2026 (UTC), `uv tool list` on a Mac that runs these builds reported `microcosm-emitter v0.13.0`.
An agent then ran `uv tool install --python 3.14 'microcosm-emitter==0.16.0'` there, and `uv tool list` reported
`microcosm-emitter v0.16.0`.

### 7. The preflight can stop the driver on exit 1 (`from the code`)

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

### 8. The QRF tail register must match the new export's concentrated columns (`predicted`)

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

### 9. If #1635 merges first, its gate shadow can stop a base or release as the tool starts (`from the code`)

[#1635](https://github.com/PolicyEngine/microcosm/pull/1635), unmerged at the pinned commit, runs each US gate's
`us/gates.json` entry in shadow beside today's gate. Every real run arms it unless `MICROCOSM_US_GATE_SHADOW=off`
([gate_shadow.py#L43-L46](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L43-L46),
[#L115-L125](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L115-L125),
read at the pull request's
[head commit](https://github.com/PolicyEngine/microcosm/commit/455d466e99540b50eea8543f0d788ce015d51191)).
On a verdict difference, today's gates set the build's exit: when a tool finishes, the shadow writes its comparison
file and then prints each difference under
`warning: US gate battery shadow for scope '<scope>' disagrees with today's gates`
([#L11-L18](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L11-L18),
[#L372-L393](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L372-L393),
[#L1013-L1043](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L1013-L1043)).
The shadow records a fault in its own evaluation, or in clearing an earlier run's files, as a shadow error, and it
warns when it cannot write its comparison
([#L504-L516](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L504-L516),
[#L613-L622](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L613-L622),
[#L1020-L1034](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L1020-L1034)).

A fault while a tool builds its shadow raises at the start. The release tool builds it once it has hashed its base and
named the release, before its first gate, and `main` re-raises the fault
([build_us_fiscal_refresh_release.py#L13806-L13839](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_fiscal_refresh_release.py#L13806-L13839),
[#L13874-L13879](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_fiscal_refresh_release.py#L13874-L13879),
[#L13426-L13452](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_fiscal_refresh_release.py#L13426-L13452)).
The driver runs the base with `--stage all --checkpoint-dir`
([route_a.sh#L522-L523](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L522-L523)),
so the base tool builds the parent's shadow before the first stage starts, and each stage builds its own
([build_us_puf_support_base.py#L893-L935](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_puf_support_base.py#L893-L935),
[#L948-L963](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_puf_support_base.py#L948-L963),
[#L1074-L1079](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_puf_support_base.py#L1074-L1079)).
A fault in a stage's setup fails that stage, and the parent stops with
`Checkpointed stage {stage!r} failed with exit code {completed.returncode}.`
([#L968-L986](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_puf_support_base.py#L968-L986)).
The driver accepts only exit 0 from the base and from the release
([route_a.sh#L534-L534](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L534-L534),
[#L585-L587](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L585-L587)).

The staged base locks its run config in the checkpoint directory before it builds the shadow, and with the switch
`off` the tool adds `gate_battery_shadow: off` to that config
([build_us_puf_support_base.py#L949-L953](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_puf_support_base.py#L949-L953),
[#L1356-L1362](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_puf_support_base.py#L1356-L1362)).
The stage runtime refuses a resume there that changes the switch, either way, with
`outer-stage run_config differs from the locked run context; start a new checkpoint directory for changed build inputs.`
([outer_stage_runtime.py#L244-L248](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/outer_stage_runtime.py#L244-L248),
[#L271-L280](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/outer_stage_runtime.py#L271-L280)).
The driver keeps those checkpoints in `$RUN/base-checkpoints`, inside the run folder for its commit
([route_a.sh#L329-L329](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L329-L329),
[#L515-L515](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L515-L515)).
To turn the shadow off for an unfinished base, as after a setup fault, an agent sets a new `RUN_ROOT` or moves
`$RUN/base-checkpoints` aside, and the base starts over
([route_a.sh#L519-L519](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L519-L519),
[outer_stage_runtime.py#L249-L256](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/outer_stage_runtime.py#L249-L256)).
The driver never exports `MICROCOSM_US_GATE_SHADOW`, so a stage sees the switch only when the run's environment or
an `export` line in the settings file carries it
([route_a.sh#L32-L34](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L32-L34),
[supervise.py#L84-L85](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/supervise.py#L84-L85)).

An agent reads the shadow's result from its comparison file: `us_national.gate_battery_shadow_comparison.json` under
`<out>/gate_battery_shadow/<release>/` for the release, and `us_base.gate_battery_shadow_comparison.json` under
`<out>/gate_battery_shadow/` for the base
([build_us_fiscal_refresh_release.py#L13264-L13280](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_fiscal_refresh_release.py#L13264-L13280),
[build_us_puf_support_base.py#L1003-L1014](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_puf_support_base.py#L1003-L1014),
[gate_shadow.py#L128-L137](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L128-L137)).
Its `verdict_difference` turns true on any difference that no threshold override explains and on any shadow error
([#L1157-L1161](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L1157-L1161),
[#L1199-L1200](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L1199-L1200)),
and the supervisor sends a stage's stderr, the warning included, to that stage's `run.log`
([supervise.py#L96-L102](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/supervise.py#L96-L102)).
When the shadow of a finished run cannot write its comparison, it prints `could not write <path>` and then each
difference under a warning that says `its comparison could not be written to <path>`
([gate_shadow.py#L1020-L1043](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L1020-L1043),
[#L372-L393](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L372-L393)).
After that write fault, an agent finds the differences only in the stage's `run.log`.
The shadow pairs today's `not_run`
with the battery's `evidence_absent` as a match, so the comparison lists an entry today's code skipped as matching,
with `not_run` in its `legacy.status`
([#L230-L236](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L230-L236),
[#L268-L270](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L268-L270),
[#L746-L758](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L746-L758)).
The driver refuses a selection source or `--exact-k` in `RELEASE_EXTRA_ARGS`, and the template's value carries no
`--input-mass-reference-h5` or `--scf-education-debt`
([route_a.sh#L241-L245](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L241-L245),
[#L272-L279](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L272-L279),
[route_a.env.example#L46-L46](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.env.example#L46-L46)).
A run on those settings records six `us_national` entries as not run
([build_us_fiscal_refresh_release.py#L7968-L7972](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_fiscal_refresh_release.py#L7968-L7972),
[#L14206-L14211](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_fiscal_refresh_release.py#L14206-L14211),
[#L15536-L15540](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_fiscal_refresh_release.py#L15536-L15540),
[#L16312-L16317](https://github.com/PolicyEngine/microcosm/blob/455d466e99540b50eea8543f0d788ce015d51191/tools/build_us_fiscal_refresh_release.py#L16312-L16317)).
