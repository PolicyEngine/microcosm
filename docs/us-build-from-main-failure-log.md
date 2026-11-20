# Failure log for a US build from main

Agents record here each gate that stops a US national build from `main`, in the order a run reaches it, so the next
attempt starts with every known stop in view. Each entry names what refuses, links its lines at the commit an agent
read, gives the message or the condition, and carries one status in its heading. `from the code` marks a refusal read in
the code and not run. `predicted` marks a stop that the code, together with a run's settings or its host, points to.
`observed` marks a stop a run hit, and the entry then names that run's commit and its log line.

## The run

Every entry cites lines at commit
[bcb5cba9d7be2b5907503878248f8ce533f044e9](https://github.com/PolicyEngine/microcosm/commit/bcb5cba9d7be2b5907503878248f8ce533f044e9),
the head of `main` at 11:24 UTC on 18 November 2026. An agent who records an earlier run names that run's commit
and links its lines there.
A run starts with `ROUTE_A_ENV=<settings file> bash tools/route_a/route_a.sh` from a checkout at that commit
([route_a.sh](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L1-L13)).
The driver hashes its inputs, checks the commit and the installed emitter, and stops before any stage on a problem
([route_a.sh#L449-L454](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L449-L454)).
It then creates a worktree at the commit, syncs it and prefetches the donors
([#L461-L498](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L461-L498),
[#L628-L650](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L628-L650)).
It builds the base from income years 2023-2025, or takes a Modal base that the hand-off accepted, and preflights it
([#L663-L716](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L663-L716)).
Last, it builds the release on the dense `--base-h5` arm, preflights the release and runs the publisher's offline
preflight, and it publishes nothing
([#L718-L773](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L718-L773)).
An agent who runs a later commit rechecks each cited line there first.

## Failures in run order

### 1. `verify_inputs` finds an input missing from `STORAGE`, or the driver cannot read a file to hash it (`predicted`)

The driver hashes six processed inputs under `STORAGE` on every run, a Modal base included: the three pooled ASEC
years, the PUF pair and ACS 2022
([route_a.sh#L134-L139](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L134-L139)).
When one of them sits outside that directory, `verify_inputs` records `$role missing: $path`
([route_a.sh#L177-L178](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L177-L178)),
and the run stops with `input problems (see above)`
([route_a.sh#L453-L453](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L453-L453)).
The driver checks the inputs again after it waits for the overnight chain
([route_a.sh#L528-L529](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L528-L529)),
and `--resolve` exits 2 on the same problems
([#L424-L424](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L424-L424)).

`sha_of` returns status 1 when `openssl` prints no digest
([route_a.sh#L158-L166](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L158-L166)),
and `verify_inputs` then records `$role could not be read to hash it: $path`
([route_a.sh#L184-L185](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L184-L185)).
On 17 November 2026 (UTC), in macOS launchd jobs, `/usr/bin/shasum`, `/usr/bin/openssl` and `/usr/bin/python3` could
not open files on an external drive (`Operation not permitted`). `sha_of` runs the first `openssl` on `PATH`. In
those jobs a Homebrew build came first and read every input on that drive, so `--resolve` reported no input
problems. The worktree's Python read a file there that `/usr/bin/python3` could not open. The driver's README offers
two ways past: put Homebrew's `bin` before `/usr/bin` on such a job's `PATH`, or copy the inputs to the internal disk
([README.md#L168-L175](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/README.md#L168-L175)).

The driver hashes the worktree's packaged crosswalk and the base H5 with the same `sha_of`, and it stops with
`could not read $W/$CROSSWALK_REL to hash it` or `could not read $BASE_H5 to hash it`
([route_a.sh#L494-L494](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L494-L494),
[#L685-L685](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L685-L685)).

### 2. The 2023-2025 pool needs `census_cps_2025.h5` and `asecpub26csv.zip` (`predicted`)

The driver pools income years 2023-2025 and writes out each year's file and pin
([route_a.sh#L101-L113](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L101-L113)).
It expects `census_cps_2025.h5`, `census_cps_2024.h5` and `census_cps_2023.h5` under `STORAGE`, and
`asecpub24csv.zip`, `asecpub25csv.zip` and `asecpub26csv.zip` under `EDU`
([route_a.sh#L134-L142](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L134-L142)).
A host set up for the 2022-2024 pool lacks `census_cps_2025.h5`, 304,753,967 bytes, and `asecpub26csv.zip`,
139,103,894 bytes, and `verify_inputs` records `$role missing: $path` for each, as in entry 1
([asec_sources.py#L106-L112](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/asec_sources.py#L106-L112),
[education_assistance_source.py#L169-L190](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/education_assistance_source.py#L169-L190)).
`tools/fetch_us_asec_sources.py` resolves each H5 against its pin and prints its path
([fetch_us_asec_sources.py#L1-L19](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/fetch_us_asec_sources.py#L1-L19)).
`fetch_asec_education_assistance_source(year, cache_dir=EDU)` downloads an archive into `EDU`, verifies it and
extracts its person file there
([README.md#L64-L67](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/README.md#L64-L67),
[education_assistance_source.py#L257-L279](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/education_assistance_source.py#L257-L279)).
The commit check refuses a commit whose `asec_sources.py` or `education_assistance_source.py` lacks one of the six
pins, and builds up to the November 2026 certified build keep the driver of their own commit
([route_a.sh#L253-L260](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L253-L260),
[#L101-L104](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L101-L104)).

The base also reads the income-2022 archive `asecpub23csv.zip` for weeks unemployed, and the driver passes no path
for it
([route_a.sh#L656-L662](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L656-L662)).
The base fetches it into `~/.cache/microcosm/cps/asec_2023`, and a download that misses its pin raises
`ASEC 2023 archive download SHA-256 mismatch: expected {expected_zip_sha256}, got {actual_digest}.`
([build_us_puf_support_base.py#L2839-L2847](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L2839-L2847),
[weeks_unemployed.py#L343-L377](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py#L343-L377)).
The prefetch fetches the same archive first, and a cached archive that matches the pin needs no download
([route_a.sh#L636-L639](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L636-L639),
[weeks_unemployed.py#L352-L388](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py#L352-L388)).

### 3. The driver needs the original SCF 2022 summary extract, which the Federal Reserve replaced (`predicted`)

`scf_wealth.py` pins two digests for the SCF 2022 summary extract, one for the zip at
`https://www.federalreserve.gov/econres/files/scfp2022s.zip` and one for its member `rscfp2022.dta`
([scf_wealth.py#L161-L176](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py#L161-L176)).
The Federal Reserve replaced that zip on 9 October 2026
([#1368](https://github.com/PolicyEngine/microcosm/issues/1368)). On 18 November 2026 (UTC), `curl -sI` of the URL
reported a length of 3,715,841 bytes and a last-modified date of 9 October 2026. The same day, a
[comment on #1368](https://github.com/PolicyEngine/microcosm/issues/1368#issuecomment-6966217460) reported the
sha256 of the replacement's member and its size, 34,487,960 bytes.

`verify_inputs` checks the file that `SCF` names against the member pin and the original member's size, 24,904,185
bytes
([route_a.sh#L148-L148](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L148-L148)).
It records `$role missing: $path` for a missing file, and `$role size $asize != $size: $path` and
`$role sha256 $actual != $sha: $path` for the replacement zip's member
([route_a.sh#L177-L187](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L177-L187)).
The run then stops with `input problems (see above)`, as in entry 1.

The prefetch stage calls `fetch_scf_2022_summary_extract()` with no arguments, so the fetch ignores `SCF` and looks
in its own cache, `~/.cache/microcosm/scf`
([route_a.sh#L628-L642](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L628-L642),
[scf_wealth.py#L521-L527](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py#L521-L527)).
Without a network call, it returns a cached `rscfp2022.dta` whose digest equals the member pin
([#L528-L533](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py#L528-L533)).
Otherwise it downloads the zip and checks the zip pin before the member pin
([#L535-L558](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py#L535-L558)).
The replacement zip, whose digest #1368 gives as `d4e139c6…`, fails that check, and the fetch raises
`SCF 2022 summary-extract zip failed sha-256 verification: expected {expected_zip_sha256}, got {digest}.`
([#L541-L547](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py#L541-L547)).
The driver logs a first failure and goes on, with
`prefetch failed (see $RUN/prefetch-sup/run.log); the base proceeds and the release retries the prefetch`
([route_a.sh#L643-L649](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L643-L649)).
After the base preflight of entry 10, the driver runs the prefetch again and stops on a second failure, before the
release starts
([#L724-L725](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L724-L725),
[#L610-L615](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L610-L615)).
It logs `FAIL: prefetch <result>; log $RUN/prefetch-sup/run.log`, with the start of the stage's `RESULT.json` as
`<result>`
([#L153-L157](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L153-L157)).
The supervisor writes the stage's output, the fetch's zip message included, to that `run.log`
([supervise.py#L96-L102](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/supervise.py#L96-L102)).

A driver run passes both checks when the file `SCF` names and the cached member match the member pin. The
release then reads `SCF` through `--scf-summary-extract` in place of the fetch
([route_a.sh#L404-L404](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L404-L404),
[build_us_fiscal_refresh_release.py#L15257-L15261](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L15257-L15261)).
A release-tool run without that flag calls the same fetch
([#L1763-L1774](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L1763-L1774)),
and on a machine whose cache lacks the original member it stops at the zip check. `--resolve` runs `verify_inputs`
but never the prefetch
([route_a.sh#L414-L425](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L414-L425)),
so it passes a run whose cache lacks the original member. The fetch checks a cached member against the member pin
alone, so the prefetch passes with a copy of the original member at `~/.cache/microcosm/scf/rscfp2022.dta`. On a
machine with no copy, the driver has no setting that passes either check, and #1368 tracks pointing the fetch at a
copy of the original zip.

### 4. A rerun from `main` must set the feed and crosswalk pins, or the driver refuses (`from the code`)

The driver keeps the published run's feed and crosswalk pins as its defaults on purpose
([route_a.sh#L75-L76](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L75-L76),
[#L115-L115](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L115-L115)).
Its comment says a rerun's settings must pin both at the chosen commit when `main` has changed them
([route_a.sh#L73-L74](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L73-L74)).
The settings template tells a rerun to set `FEED_SHA`, `FEED_BYTES` and `CROSSWALK_SHA` beside `FEED`, after
checking that commit's packaged pins, and it adds that `--resolve` names both of the commit's digests
([route_a.env.example#L21-L24](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.env.example#L21-L24)).
The driver's test pins that choice, with the published digests when the settings leave them unset and the settings'
values when they set them
([test_route_a_driver.py#L109-L145](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/tests/engine_free/us/test_route_a_driver.py#L109-L145)).

At the pinned commit, `us/chronicle_feed.json` pins `facts_sha256` `1d45a89c…`
([chronicle_feed.json#L14-L14](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json#L14-L14)),
and the packaged crosswalk hashes to `a347303f…`, so the driver refuses a run from that commit that keeps the
defaults. With the template's `FEED` and the default pins, `check_commit` records these two blockers
([route_a.sh#L244-L251](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L244-L251),
[#L261-L271](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L261-L271)):

```
us/chronicle_feed.json at $full does not pin facts_sha256 $FEED_SHA (the release tool would refuse the feed); it pins ${pinned_feed:-no readable facts_sha256}: set FEED to that feed's file, FEED_SHA to its digest and FEED_BYTES to its size
packaged CD vintage crosswalk at $full is $crosswalk, not $CROSSWALK_SHA: set CROSSWALK_SHA to it once the change is reviewed
```

The first names the feed digest the commit pins, and the second names the crosswalk's digest at the commit.
A settings file that points `FEED` at main's feed but keeps the default digest and size also fails `verify_inputs`
on that feed's size and digest
([route_a.sh#L181-L187](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L181-L187)).

The driver runs `verify_inputs` and then `check_commit`, and it logs every problem before it stops
([route_a.sh#L449-L454](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L449-L454)).
A run exits on `input problems (see above)` first and on `commit blockers (see above)` next, while `--resolve` exits
2 on input problems and only logs commit blockers
([route_a.sh#L420-L425](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L420-L425)).
The 18 October rerun runbook's first script stops sooner, with
`REFUSED: main changed the release feed; review its pins/settings`
([us-route-a-rerun-20261018.md#L199-L201](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-route-a-rerun-20261018.md#L199-L201)).

### 5. An emitter older than the 0.16.0 minimum stops a run that requires telemetry (`from the code`)

A build starts the separately installed `microcosm-emitter` as its child process, for telemetry, Orrery graph
publication and the Hugging Face hand-over of staging run files
([telemetry_emitter.py#L1-L8](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L1-L8)).
The adapter never starts an emitter older than the minimum that `minimum_emitter_version.json` sets
([telemetry_emitter.py#L35-L38](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L35-L38),
[minimum_emitter_version.json#L2-L2](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/minimum_emitter_version.json#L2-L2)).

The driver checks the emitter at the end of its commit check, before it creates a worktree
([route_a.sh#L320-L320](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L320-L320),
[#L323-L341](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L323-L341)).
`check_emitter.py` repeats the release's checks against the commit's minimum without importing `microcosm`
([check_emitter.py#L1-L24](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/check_emitter.py#L1-L24)).
For an older emitter, it reports
([check_emitter.py#L139-L144](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/check_emitter.py#L139-L144)):

```
microcosm-emitter {version} at {command} is older than the commit's floor {floor} (uv tool install --upgrade 'microcosm-emitter>={floor}', or point {COMMAND_VARIABLE} at a newer one)
```

With `MICROCOSM_TELEMETRY_REQUIRED=1`, the check reports that problem as a blocker, adding that the release would stop
before any work, and the run stops with `commit blockers (see above)`
([check_emitter.py#L177-L181](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/check_emitter.py#L177-L181),
[route_a.sh#L454-L454](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L454-L454)).
Unset or `0`, the check reports a note, and the release runs without collector telemetry or staging hand-overs
([check_emitter.py#L182-L185](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/check_emitter.py#L182-L185)).
`--resolve` logs the blocker and still exits 0 when the inputs pass
([route_a.sh#L420-L425](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L420-L425)).

The driver exports `MICROCOSM_TELEMETRY_REQUIRED` and `MICROCOSM_EMITTER_COMMAND` when the settings file sets them,
so the check and the stages see the same values
([route_a.sh#L62-L65](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L62-L65)).
The supervisor passes the caller's environment through to each stage
([supervise.py#L84-L85](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/supervise.py#L84-L85)).
The template leaves the requirement commented out and says PolicyEngine builds set it to `1`
([route_a.env.example#L49-L55](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.env.example#L49-L55)).
The module and the README both say PolicyEngine's own builds keep telemetry required
([telemetry_emitter.py#L31-L33](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L31-L33),
[README.md#L139-L141](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/README.md#L139-L141)).

Inside a stage, with the requirement on, an emitter the adapter cannot use stops the build before it does any work
([telemetry_emitter.py#L24-L28](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L24-L28),
[#L386-L401](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L386-L401)).
For an older emitter, the adapter raises `TelemetryRequiredError` with reason `outdated_emitter`, saying the
installed emitter
`reports version {version}, and this build needs {MINIMUM_EMITTER_VERSION} or later`
([telemetry_emitter.py#L444-L461](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L444-L461),
[#L572-L573](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L572-L573)).
A local base starts the first emitter after parsing its arguments and before reading any input
([build_us_puf_support_base.py#L8-L12](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L8-L12),
[#L879-L884](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L879-L884)).
After a Modal base, the release starts the first emitter once it has parsed its arguments
([build_us_fiscal_refresh_release.py#L13718-L13730](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L13718-L13730)).
Unset or `0`, a missing, refusing or outdated emitter leaves the build running without it
([telemetry_emitter.py#L14-L15](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L14-L15)).
For an older emitter the adapter prints a warning, and the build continues without hosted telemetry or staging
hand-overs and queues no Orrery graph publication
([#L192-L202](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L192-L202),
[#L574-L583](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/telemetry_emitter.py#L574-L583)).

On 18 November 2026 (UTC), `uv tool list` on a Mac that runs these builds reported `microcosm-emitter v0.13.0`.
An agent then ran `uv tool install --python 3.14 'microcosm-emitter==0.16.0'` there, and `uv tool list` reported
`microcosm-emitter v0.16.0`.

### 6. A rerun whose worktree synced with another interpreter stops at the sync marker (`from the code`)

`ensure_worktree` syncs a fresh worktree with `"$UV" sync --python "$UV_SYNC_PYTHON" --all-packages --locked --extra us`
and unsets `UV_FROZEN` inside the sync
([route_a.sh#L471-L476](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L471-L476)).
It stops with `uv sync in $WT` when the sync fails, and after a sync that completes it records the interpreter
setting in a marker
([#L476-L478](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L476-L478)).
`UV_SYNC_PYTHON` defaults to `3.14`, because the locked torch 2.12.0 carries wheels for CPython 3.13 and 3.14 and no
source archive
([route_a.sh#L48-L50](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L48-L50),
[uv.lock#L2319-L2357](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/uv.lock#L2319-L2357)).
The driver names each worktree after the first 12 characters of its commit, so a rerun of that commit finds the
marker
([route_a.sh#L461-L470](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L461-L470)).
When the marker records another setting, the driver stops with
`worktree $WT was synced with --python $synced, not UV_SYNC_PYTHON=$UV_SYNC_PYTHON: set it back, or remove $marker to re-sync`
([#L479-L488](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L479-L488)).
A marker that records no setting keeps its venv, and the driver logs
`worktree $WT was synced before its interpreter was recorded; keeping its venv`
([#L483-L484](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L483-L484)).

On 17 November 2026 (UTC) an agent ran the sync command of commit
[c7a5f9679](https://github.com/PolicyEngine/microcosm/blob/c7a5f96796ac6a16e7c520925e2517e9abbc8512/tools/route_a/route_a.sh#L344-L346),
`uv sync --all-packages --locked --extra us`, outside the driver, with `UV_PYTHON=3.14` and without `UV_FROZEN`, and
it completed. Neither `pyproject.toml` nor `uv.lock` changed between that commit and the pinned one.

### 7. The base needs 80 GiB of available RAM locally, or a lineage match on Modal (`predicted`)

The driver admits a local base only with 80 GiB of available RAM and, before any checkpoint exists, 70 GiB of
free disk, and it caps the base at 100 GiB RSS
([route_a.sh#L666-L681](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L666-L681)).
`wait_resources` waits up to `RESOURCE_WAIT_HOURS`, 12 by default, and then fails with
`base: needs 80 GiB available RAM and <n> GiB free disk`
([route_a.sh#L47-L47](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L47-L47),
[#L533-L540](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L533-L540)).
The driver's comment records 44 GB of frame checkpoints from a base measured on 16 September under policyengine-us
1.819.0
([route_a.sh#L652-L655](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L652-L655)).
It removes them after hashing the base only with `PRUNE_BASE_CHECKPOINTS=1`, which the template sets
([route_a.sh#L691-L694](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L691-L694),
[route_a.env.example#L39-L39](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.env.example#L39-L39)).

The driver skips the base stage when the run folder holds `base-sup/ACCEPTED`
([route_a.sh#L572-L572](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L572-L572),
[#L666-L666](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L666-L666)),
which the Modal hand-off writes last
([README.md#L275-L284](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/README.md#L275-L284)).
For the 2023-2025 pool, the Modal template names the registration `us-puf-support-base-income-2023-2025` and twelve
inputs, the weeks-unemployed archive among them
([us-modal-stage-route-a-rerun-base-plan.json#L1-L58](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-modal-stage-route-a-rerun-base-plan.json#L1-L58),
[modal_us_stage_plan.py#L1129-L1157](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/modal_us_stage_plan.py#L1129-L1157)).
Its `source.commit` holds the placeholder `PICK_MAIN_COMMIT_FOR_THE_NEXT_CERTIFIED_BUILD`, and the plan tool refuses
it with `source.commit must be a full 40-hex lowercase sha`
([plan.json#L6-L9](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-modal-stage-route-a-rerun-base-plan.json#L6-L9),
[modal_us_stage_plan.py#L1515-L1517](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/modal_us_stage_plan.py#L1515-L1517)).

The hand-off's `compare-lineage` requires the Modal run to match a local reference, source digest included
([us-modal-stage-runbook.md#L1290-L1314](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-modal-stage-runbook.md#L1290-L1314)).
The committed reference records a local run of the tool `us-puf-support-base` at an older commit
([reference#L4-L6](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-modal-stage-route-a-base-local-reference.json#L4-L6),
[#L216-L216](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-modal-stage-route-a-base-local-reference.json#L216-L216);
[README.md#L291-L296](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/README.md#L291-L296)).
`compare-lineage` reports a receipt whose tool or commit differs from the reference's
([modal_us_stage_plan.py#L4207-L4217](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/modal_us_stage_plan.py#L4207-L4217)),
so a Modal base at the pinned commit needs a local reference made at that commit first.
`make_local_reference.py` makes one from a filled Route A base plan and the checkpoint directory of a local run that
completed stages 000 and 001 and no others
([make_local_reference.py#L118-L129](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/make_local_reference.py#L118-L129)).
It refuses an input of that run whose file name or digest differs from the plan's
([#L186-L192](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/make_local_reference.py#L186-L192)).
The Modal template pins the published run's feed, `b8543739…`, as the base's ledger facts
([plan.json#L51-L54](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-modal-stage-route-a-rerun-base-plan.json#L51-L54)).
A local run on main's feed of entry 4 fails that check until the run and the plan name the same feed.
The driver hands the base `BASE_LEDGER_FACTS`, which defaults to `FEED`, so a settings file can set it and
`BASE_LEDGER_FACTS_SHA` to the published run's feed and digest while the release reads main's
([route_a.sh#L89-L92](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L89-L92),
[#L378-L381](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L378-L381)).
When the stage-001 checkpoints differ, `lineage_class_check.py` exits 1 on any differing column other than
`person.pre_subsidy_rent`, even inside its 1e-12 relative tolerance
([lineage_class_check.py#L27-L28](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/lineage_class_check.py#L27-L28),
[#L112-L122](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/lineage_class_check.py#L112-L122),
[#L138-L138](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/lineage_class_check.py#L138-L138)).

### 8. The gate shadow can stop a base or release as the tool starts (`from the code`)

The pinned commit merges [#1635](https://github.com/PolicyEngine/microcosm/pull/1635), which runs each US gate's
`us/gates.json` entry in shadow beside today's gate. Every real run arms it unless `MICROCOSM_US_GATE_SHADOW=off`
([gate_shadow.py#L43-L46](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L43-L46),
[#L115-L125](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L115-L125)).
On a verdict difference, today's gates set the build's exit: when a tool finishes, the shadow writes its comparison
file and then prints each difference under
`warning: US gate battery shadow for scope '<scope>' disagrees with today's gates`
([#L11-L18](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L11-L18),
[#L372-L393](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L372-L393),
[#L1013-L1043](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L1013-L1043)).
The shadow records a fault in its own evaluation, or in clearing an earlier run's files, as a shadow error, and it
warns when it cannot write its comparison
([#L504-L516](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L504-L516),
[#L613-L622](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L613-L622),
[#L1020-L1034](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L1020-L1034)).

A fault while a tool builds its shadow raises at the start. The release tool builds it once it has hashed its base and
named the release, before its first gate, and `main` re-raises the fault
([build_us_fiscal_refresh_release.py#L13806-L13839](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L13806-L13839),
[#L13874-L13879](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L13874-L13879),
[#L13426-L13452](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L13426-L13452)).
`main` also re-raises a fault in the shadow's wiring when the shadow finishes after a release, while a difference
alone raises nothing
([#L13453-L13468](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L13453-L13468)).
The driver runs the base with `--stage all --checkpoint-dir`
([route_a.sh#L671-L672](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L671-L672)),
so the base tool builds the parent's shadow before the first stage starts, and each stage builds its own
([build_us_puf_support_base.py#L893-L935](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L893-L935),
[#L948-L963](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L948-L963),
[#L1074-L1079](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L1074-L1079)).
A fault in a stage's setup fails that stage, and the parent stops with
`Checkpointed stage {stage!r} failed with exit code {completed.returncode}.`
([#L968-L986](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L968-L986)).
The driver accepts only exit 0 from the base and from the release
([route_a.sh#L681-L681](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L681-L681),
[#L734-L736](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L734-L736)).

The staged base locks its run config in the checkpoint directory before it builds the shadow, and with the switch
`off` the tool adds `gate_battery_shadow: off` to that config
([build_us_puf_support_base.py#L949-L953](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L949-L953),
[#L1356-L1362](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L1356-L1362)).
The stage runtime refuses a resume there that changes the switch, either way, with
`outer-stage run_config differs from the locked run context; start a new checkpoint directory for changed build inputs.`
([outer_stage_runtime.py#L244-L248](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/outer_stage_runtime.py#L244-L248),
[#L271-L280](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/outer_stage_runtime.py#L271-L280)).
The driver keeps those checkpoints in `$RUN/base-checkpoints`, inside the run folder for its commit
([route_a.sh#L456-L456](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L456-L456),
[#L663-L663](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L663-L663)).
To turn the shadow off for an unfinished base, as after a setup fault, an agent sets a new `RUN_ROOT` or moves
`$RUN/base-checkpoints` aside, and the base starts over
([route_a.sh#L667-L667](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L667-L667),
[outer_stage_runtime.py#L249-L256](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/outer_stage_runtime.py#L249-L256)).
The driver exports only the two emitter settings of entry 5, so a stage sees `MICROCOSM_US_GATE_SHADOW` only when the
run's environment or an `export` line in the settings file carries it
([route_a.sh#L32-L34](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L32-L34),
[#L62-L65](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L62-L65),
[supervise.py#L84-L85](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/supervise.py#L84-L85)).

An agent reads the shadow's result from its comparison file: `us_national.gate_battery_shadow_comparison.json` under
`<out>/gate_battery_shadow/<release>/` for the release, and `us_base.gate_battery_shadow_comparison.json` under
`<out>/gate_battery_shadow/` for the base
([build_us_fiscal_refresh_release.py#L13264-L13280](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L13264-L13280),
[build_us_puf_support_base.py#L1003-L1014](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L1003-L1014),
[gate_shadow.py#L128-L137](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L128-L137)).
Its `verdict_difference` turns true on any difference that no threshold override explains and on any shadow error
([#L1157-L1161](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L1157-L1161),
[#L1199-L1200](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L1199-L1200)),
and the supervisor sends a stage's stderr, the warning included, to that stage's `run.log`
([supervise.py#L96-L102](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/supervise.py#L96-L102)).
When the shadow of a finished run cannot write its comparison, it prints `could not write <path>` and then each
difference under a warning that says `its comparison could not be written to <path>`
([gate_shadow.py#L1020-L1043](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L1020-L1043),
[#L372-L393](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L372-L393)).
After that write fault, an agent finds the differences only in the stage's `run.log`.
The shadow pairs today's `not_run`
with the battery's `evidence_absent` as a match, so the comparison lists an entry today's code skipped as matching,
with `not_run` in its `legacy.status`
([#L230-L236](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L230-L236),
[#L268-L270](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L268-L270),
[#L746-L758](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L746-L758)).
The driver refuses a selection source or `--exact-k` in `RELEASE_EXTRA_ARGS`, and neither its release command nor the
template's value carries `--input-mass-reference-h5` or `--scf-education-debt`
([route_a.sh#L343-L358](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L343-L358),
[#L399-L406](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L399-L406),
[route_a.env.example#L65-L65](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.env.example#L65-L65)).
A run on those settings records six `us_national` entries as not run
([build_us_fiscal_refresh_release.py#L7968-L7972](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L7968-L7972),
[#L14206-L14211](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L14206-L14211),
[#L15536-L15540](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L15536-L15540),
[#L16312-L16317](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L16312-L16317)).

### 9. The Modal hand-off's copy stops on the base's gate-shadow directory (`predicted`)

A base at the pinned commit writes the shadow's `gate_battery_shadow/` directory inside its `--out`, and a Modal base
writes to `state/base-out`
([build_us_puf_support_base.py#L1003-L1014](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_puf_support_base.py#L1003-L1014),
[gate_shadow.py#L106-L106](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/gate_shadow.py#L106-L106),
[modal_us_stage_plan.py#L1026-L1027](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/modal_us_stage_plan.py#L1026-L1027)).
The Modal runner lists every regular file under the state tree in the receipt, files in subdirectories included
([modal_us_stage.py#L1117-L1135](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/modal_us_stage.py#L1117-L1135),
[modal_us_stage_plan.py#L2963-L2983](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/modal_us_stage_plan.py#L2963-L2983),
[#L3181-L3208](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/modal_us_stage_plan.py#L3181-L3208)).
An agent runs the hand-off after the Modal base finishes and before the driver reaches its base stage, which then
skips on `base-sup/ACCEPTED` as in entry 7
([README.md#L275-L284](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/README.md#L275-L284)).

The hand-off block runs in a subshell under `set -euo pipefail` and copies the base with
`cp -p ./modal-runs/$R/state/base-out/* "$RUN/base-out/"`, which has no `-R`
([us-modal-stage-runbook.md#L1379-L1391](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-modal-stage-runbook.md#L1379-L1391)).
On 18 November 2026 (UTC), macOS `cp -p` given a directory operand without `-R` printed
`cp: <directory> is a directory (not copied).` and exited 1. The block then ends at the copy, before `verify-receipt`,
`base.sha256` and `ACCEPTED`
([#L1393-L1411](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-modal-stage-runbook.md#L1393-L1411)).
The runbook tells an agent to remove what a stopped block wrote and run it again, and the rerun stops at the same
copy
([#L1372-L1375](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-modal-stage-runbook.md#L1372-L1375)).
A copy without the directory would also fail `verify-receipt`, which reports `missing: {rel}` for each output the
receipt lists under `base-out` and the run folder lacks
([modal_us_stage_plan.py#L3390-L3404](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/modal_us_stage_plan.py#L3390-L3404),
[#L3332-L3340](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/modal_us_stage_plan.py#L3332-L3340)).
The 18 October rerun runbook copies the same way, with `cp -pc`
([us-route-a-rerun-20261018.md#L460-L471](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-route-a-rerun-20261018.md#L460-L471)).
[#1677](https://github.com/PolicyEngine/microcosm/pull/1677), an open draft on 18 November 2026, changes the two copies
to `cp -pR` and `cp -pRc`.

Once the copy passes, as with `cp -pR`, the block hashes the base with `shasum` inside an `echo` argument
([us-modal-stage-runbook.md#L1393-L1396](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-modal-stage-runbook.md#L1393-L1396)).
When the Python of `verify-receipt` reads the base and `shasum` cannot, as in the launchd jobs of entry 1, `set -e`
lets the empty digest through.
The driver then stops with `$RUN/base.sha256 holds no sha256; delete it and rerun`
([route_a.sh#L684-L689](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L684-L689)).

### 10. The preflight can stop the driver on exit 1 (`from the code`)

The driver runs `tools/preflight_us_release_gates.py` on the base with `--ledger-facts "$FEED"`, and it accepts exit 0
(clean) or 2 (at risk) unless `RELEASE_DESPITE_PREFLIGHT_FAIL=1`
([route_a.sh#L699-L716](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L699-L716)).
The release preflight accepts only exit 0 or 2, whatever `RELEASE_DESPITE_PREFLIGHT_FAIL` says
([#L751-L756](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L751-L756)).
Given `--ledger-facts`, the preflight compiles the fiscal-target registry the release calibrates, with age targets
and the packaged crosswalk, and previews zero support on it beside its export-mass, smoke-support and SPM
composition checks
([preflight_us_release_gates.py#L308-L315](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/preflight_us_release_gates.py#L308-L315),
[#L373-L383](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/preflight_us_release_gates.py#L373-L383),
[release_gate_preflight.py#L943-L977](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/release_gate_preflight.py#L943-L977),
[#L1063-L1074](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/us_runtime/release_gate_preflight.py#L1063-L1074)).
The feed's re-pin record lists one compile of main's feed, 33,250 targets with 6,102 on the `national_state` surface
([us-chronicle-feed-repin.md#L170-L186](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/docs/us-chronicle-feed-repin.md#L170-L186)).

On 17 November 2026 (UTC) an agent ran the preflight outside the driver, offline, at commit
[c7a5f9679](https://github.com/PolicyEngine/microcosm/commit/c7a5f96796ac6a16e7c520925e2517e9abbc8512), on an
existing base built at 9434c504b, with `--new-lineage`, and it exited 2, which the driver accepts. It skipped
selection carry-over and passed its capital-gains tail, smoke-support and SPM composition checks. Its zero-support
preview could statically check none of the 33,193 positive targets, so it passed without grading any. Its
export-mass check found two columns under their bands at base weights.
`partnership_income` held $394.2 billion against a band of $423.9 billion to $1,271.6 billion, and
`social_security_survivors` held $67.8 billion against $80.5 billion to $241.5 billion. The preflight asks an agent
to confirm that calibration targets each column; this run measured neither column after calibration.
Neither `preflight_us_release_gates.py` nor `release_gate_preflight.py` changed between that commit and the pinned
one. That base pooled income years 2022-2024
([route_a.sh at 9434c504b#L524-L524](https://github.com/PolicyEngine/microcosm/blob/9434c504ba2a5723ea4a88feab57ef67000ad9dc/tools/route_a/route_a.sh#L524-L524)),
and a base at the pinned commit pools 2023-2025
([route_a.sh#L673-L677](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L673-L677)).

### 11. The QRF tail register must match the new export's concentrated columns (`predicted`)

At its terminal batch, after calibration, the release evaluates the QRF tail-concentration gate on the export and
raises `Release gates failed: ...` on any failure in the batch
([build_us_fiscal_refresh_release.py#L17410-L17474](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L17410-L17474)).
A sparse QRF-imputed column whose top 100 weighted records carry over 75% of its mass fails unless the register
lists it, and a listed column at or under that share fails as stale
([gates.py#L3370-L3372](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/gates.py#L3370-L3372),
[#L3436-L3457](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/packages/microcosm-build/src/microcosm/build/gates.py#L3436-L3457)).
A listed column that the gate found stale or never checked adds
`QRF tail-concentration register mismatch: the per-run exclusion register must exactly match the concentrated columns`
to the batch, whatever `--allow-qrf-tail-concentration` says
([build_us_fiscal_refresh_release.py#L8598-L8631](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L8598-L8631),
[#L8658-L8671](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/build_us_fiscal_refresh_release.py#L8658-L8671)),
and the driver refuses any `--allow-*` flag in `RELEASE_EXTRA_ARGS`
([route_a.sh#L343-L358](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L343-L358)).
The template's default register comes from the published run
([route_a.env.example#L66-L67](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.env.example#L66-L67),
[README.md#L152-L154](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/README.md#L152-L154)),
and the driver's comment asks for a register measured on the export that ships
([route_a.sh#L95-L98](https://github.com/PolicyEngine/microcosm/blob/bcb5cba9d7be2b5907503878248f8ce533f044e9/tools/route_a/route_a.sh#L95-L98)).
A base built at the pinned commit, from a new pool of years, brings its own concentrated columns, so a register from
another export will likely fail here.
