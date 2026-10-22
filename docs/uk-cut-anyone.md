# Building a UK national cut without the original builder's machine

Any team member or agent session with the credentials below can build, sign
and publish a Microcosm UK national cut, then open its policyengine.py
certification pull request. Each input comes from Hugging Face and is checked
against the pins the build itself enforces.

This is the short path. The step-by-step reference behind it stays in
[uk-national-release-assembly-runbook-806.md](uk-national-release-assembly-runbook-806.md).
The UK hub's runbook of the hand-run build is at
`~/reviews/uk-hub/jobs/microcosm-cut-runbook/RUNBOOK.md` on Max's machine.

## What you need

| Need | Where it comes from | Check |
|---|---|---|
| A clean checkout of `main` locked at the policyengine-uk you want | a "Lock policyengine-uk X" PR merged to main (runbook §2) | `tools/run_uk_cut.py preflight` |
| `HF_TOKEN` | write access to `policyengine/populace-uk-private` and `policyengine/populace-uk-staging`, and read access to the input repos below | preflight |
| `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY` | a key registered **active** in `microcosm.data.gate_signing_keys` (see [Signing keys](#signing-keys)) | `python -m microcosm.data.gate_signing_keys check` |
| About 60 GB of free disk, AC power, a quiet host | the machine | preflight |

## One command

```bash
cd <clean checkout of main>
uv sync --all-packages --locked --extra uk
export HF_TOKEN=…   MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY=…
uv run --no-sync python tools/run_uk_cut.py all --label sat24              # build, certify, assemble
uv run --no-sync python tools/run_uk_cut.py all --label sat24 --publish    # and publish for inspection
```

Agent sessions on Max's build Mac read the team key with
`agent-secret get microcosm/uk-gate-signing-key-team maxghenis`. The key can
sign once #1350 registers it, after María agrees. `HF_TOKEN` comes from
`agent-secret get HUGGING_FACE_TOKEN_MAX maxghenis`.

`all` runs these steps in order:

1. `fetch`
2. `preflight`
3. spine
4. national dry run, then the national calibration
5. re-score, only when the in-build evaluation did not pass
6. certify
7. stage-1 comparison (with `--previous-h5`)
8. assemble
9. publish preflight
10. publish (with `--publish`)

Every step first checks whether its outputs already exist and verify, and
skips itself if they do. So when a run stops, fix the cause and run the same
command again.

Nothing is overwritten:

- A half-built spine is left for you to inspect. Rebuilding at the same path
  would delete it, so pick a new `--label` instead.
- An unfinished national run is moved aside only with `--retry-national`.
- An existing release directory is never re-assembled.

Other subcommands:

- `status --label L`: where each step stands.
- `fetch`: install and verify the inputs only.
- `preflight --label L`: every check, with nothing built.
- `inputs`: the resolved pins.

Outputs go to `~/microcosm-uk-work`, outside any repository, because they
derive from licensed data:

| Path | Contents |
|---|---|
| `inputs/` | shared verified inputs |
| `cuts/<label>-<sha9>/spine/` | the spine |
| `cuts/<label>-<sha9>/national/` | the calibration run |
| `cuts/<label>-<sha9>/release/` | the assembled release |
| `cuts/<label>-<sha9>/logs/` | one log per step |
| `cuts/<label>-<sha9>/cut_receipt.json` | commit, engine, key fingerprint, input pins, step timings, tag |

On Max's machine, protect the work directory from disk-guard for the length
of a run (runbook §1).

### What it will not do

- **It never moves a default.** `--publish` always passes `--no-latest
  --tag-only`, so `populace-uk-private`'s main branch, its root files and
  `latest-*.json` stay where they are (runbook Q12 and F14).
- **It never promotes.** Promotion (`--promote-line national`) is a separate,
  reviewed call. The runner prints the command for it.
- **It never certifies into policyengine.py.** That is a pull request in
  another repository; see [Certifying into policyengine.py](#certifying-into-policyenginepy).
- **It builds only from main.** It refuses a commit that is not on
  `origin/main`, unless you pass `--allow-off-main "<who agreed and why>"` for
  a pushed commit.

## Inputs

`tools/uk_cut_inputs.json` says only *where* each input lives. Every expected
digest is read at run time from the pins the build already checks:
`uk/source_stages.json`, `uk/frs_release.json`, `uk/chronicle_feed.json`,
`uk/gates.json` and `weighted_integrity.py`. A file is installed only if it
is byte-identical to what the build will accept, and nothing unverified is
ever left at an input's path.

| Input | Hosted at | State |
|---|---|---|
| FRS 2024-25 (14 tabs) | `policyengine/policyengine-uk-data-private` `frs_2024_25.zip` @ `a2039519` (gated) | fetched, every member matches its pin |
| SPI 2022-23, WAS round 8, ETB 1977-2024 | the same repo's `spi_2022_23.zip`, `was_2006_22.zip`, `etb_1977_24.zip` | fetched, matches |
| HMRC personal incomes ODS | gov.uk URL pinned in `source_stages.json` | fetched, matches |
| Incumbent enhanced FRS | `policyengine-uk-data-private` `enhanced_frs_2024_25.h5` @ `a9e52499` | fetched, matches |
| Input-mass reference | `policyengine/microcosm-uk-build-inputs` (private) `input-mass-reference/fd41cb5f…/` | fetched, matches |
| Chronicle UK consumer artifact (7.4 GB) | `policyengine/microcosm-uk-build-inputs` `chronicle-uk/abf0921b…/` | rebuilt on Modal; hosted only if it reproduces the pin |
| NTS 2002-2024 (5 tabs) | `policyengine-uk-data-private` `nts_2002_24.zip` | **needs a one-time upload** |
| LCFS 2024-25 (2 tabs) | `policyengine-uk-data-private` `lcfs_2024_25.zip` | **needs a one-time upload** |

### Uploading NTS and LCFS (once, by a UKDS licence holder)

Upload the UK Data Service downloads unchanged: the TAB-format zips of SN 5340
(19th edition) and SN 9603. The runner finds each tab by name inside the zip
and checks it against its pin, so the folder layout inside does not matter.

```bash
hf upload policyengine/policyengine-uk-data-private ./<SN5340 tab zip> nts_2002_24.zip --repo-type model \
  --commit-message "Add NTS 2002-2024 (UKDS SN 5340 ed. 19, TAB) for Microcosm UK builds"
hf upload policyengine/policyengine-uk-data-private ./<SN9603 tab zip> lcfs_2024_25.zip --repo-type model \
  --commit-message "Add LCFS 2024-25 (UKDS SN 9603, TAB) for Microcosm UK builds"
```

Then confirm, without building anything:

```bash
uv run --no-sync python tools/run_uk_cut.py fetch --only nts_2002_2024 --only lcfs_2024_25
```

A tab that differs from its pin is refused by name. If that happens, the
upload is a different edition from the one the build was reviewed against.
Afterwards, pin the upload's revision and archive sha256 in
`tools/uk_cut_inputs.json`, in a small follow-up PR.

### Rebuilding the Chronicle artifact

The UK feed is built from the public Chronicle repository. It reads its
committed source files plus two DESNZ workbooks, which it fetches from the
publisher by pinned sha256 (`LEDGER_SOURCE_ARTIFACT_FETCH=1`). No R2
credentials are needed. The bundle needs tens of gigabytes, so it runs on
Modal:

```bash
modal run tools/modal_rebuild_uk_chronicle_artifact.py                                       # plan, free
HF_TOKEN=… modal run --detach tools/modal_rebuild_uk_chronicle_artifact.py --build --upload  # rebuild
```

The artifact is uploaded under `chronicle-uk/<commit>/` only when both of its
digests match `chronicle_feed.json`. Otherwise only the receipt and the bundle
reports go to `diagnostic/`, so the difference can be traced. The token is
passed for that run only and is never stored as a Modal secret.

### Regenerating the input-mass reference

The certifier binds the reference by a canonical digest of its identity and
totals. The emitted column set follows the installed policyengine-uk.
Today's engine adds `bus_fare_spending` and `sic_industry_division`. That
changes the digest, though all 131 shared totals are identical. Reproduce the
pinned file (`fd41cb5f…`) at the commit that pinned it:

```bash
git worktree add --detach ../mc-0eac9bbbd 0eac9bbbd && cd ../mc-0eac9bbbd
uv sync --all-packages --locked --extra uk        # policyengine-uk 2.89.0
uv run --no-sync python tools/build_uk_efrs_parity_reference.py \
  --input-h5 <enhanced_frs_2024_25.h5 @ a9e52499> --emit-weighted-totals <outside the repo>.json
```

The hosted copy's receipt records this recipe.

## Signing keys

Gate reports, release-cut certifications and dense gate reports are signed
with HMAC-SHA256. Verification now also requires the key's fingerprint to be
registered in `microcosm.data.gate_signing_keys`. Before this, any key that
signed a report also verified it.

Each registered key has a status:

- **active**: signs and verifies.
- **retired**: verifies the cuts it already signed, but signs nothing new.
- **revoked**: does neither.

Every UK signer refuses an unregistered or retired key:

- the gate battery;
- the certifier;
- the scoped re-signer;
- the national role, which checks before it loads any input. Before, a
  missing key failed only after the solve (runbook F8).

HMAC verification needs the secret that signed. So a cut is signed,
certified and published by one key holder, and the registry holds one key
per builder rather than one shared secret.

| Label | Fingerprint | Held by | Status |
|---|---|---|---|
| `uk-gate-juaristi22-2026` | `66bf0e29…` | María Juaristi's build machine | active; signed every cut on record |
| `uk-gate-team-2026` | `dd31e325…` | agent-secret `microcosm/uk-gate-signing-key-team` on Max's build Mac | proposed in #1350; registers when María agrees |

**Add a builder or rotate a key.**

1. Run `python -m microcosm.data.gate_signing_keys generate`. It prints the
   key on stdout and the fingerprint on stderr.
2. Store the key in a secret store. Never commit or print it.
3. Open a pull request that adds the fingerprint as `active`.
4. To rotate, set the old entry to `retired` with a `retired_on` date, in
   the same or a later PR. Its cuts still verify, but only for someone who
   holds that key.

**Revoke a leaked key.** Set it to `revoked`. Every cut it signed then fails
verification, so re-sign or withdraw those cuts.

## Certifying into policyengine.py

```bash
uv run --no-sync python tools/certify_uk_cut_in_policyengine_py.py \
  --national-tag microcosm-uk-2024-25-national-<ts>-<id> \
  (--dense-tag <tag> | --keep-regional-release | --area-coded-national-uri <uri> | --drop-regional-release) \
  [--push --open-pr]
```

The script does the mechanical part of what policyengine.py#685 did by hand,
on a new branch from policyengine.py's main.

**Before writing anything:**

- It refuses if an open pull request or branch is already certifying the cut
  (one writer per cut).
- It reads both manifests at their tags.
- It refuses a regional cut built on another policyengine-uk, naming the
  routes.

**Then:**

1. `certify-data` at the cut's own model and core.
2. `uv lock` and `release_lock.py`.
3. `generate --include-tros`.
4. policyengine.py's own certified-constant rewrite, plus the local-area tag.
5. The household snapshots.
6. `check --published-spm --include-tros --strict-tros`.
7. The UC floor.
8. The tests.

It commits. It pushes and opens a draft only when asked. The pull request
body lists what needs a person: the UC floor acceptance, any dataset overlays
certification dropped, warnings and failing tests.

## Regions: the dense cut

policyengine.py refuses to route constituency and local-authority regions to
a dense cut that does not cover the certified policyengine-uk. The only dense
cut is on 2.132.6. A dense build takes 2–13 hours, and its two-build sign-off
last took about 30 hours, so a new dense cut per re-cut is not workable
before the Budget.

There are four routes:

1. **Ship together.** Build a new dense cut on the same version. This is
   correct but slow, and it needs the atomic-support files that only María
   holds.
2. **Accept the older dense cut, recorded as a mismatch.** This needs a
   reviewed policyengine.py change (the `--accept-regional-model-mismatch`
   draft).
   - Area receipts are already uncertified: the certificate covers the
     national dataset only (policyengine.py#685).
   - policyengine-core loads an older file under a newer model. It drops
     columns the model no longer defines and warns. So inputs added since
     2.132.6 default in area runs, and the mismatch record must list them.
3. **Drop routing.** sim-api then cannot serve the UK default's constituency
   and local-authority runs.
4. **An enhanced-FRS release that claims the version.** This needs a new
   uk-data release on Max's go.

The choice is María's (runbook Q13). The recommendation and its evidence are
in microcosm#1348.
