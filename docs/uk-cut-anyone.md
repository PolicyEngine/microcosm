# Building a UK national cut without the original builder's machine

Any team member or agent session with the credentials below can build, sign
and publish a Microcosm UK national cut, then open its policyengine.py
certification pull request. Each input comes from Hugging Face and is checked
against the pins the build itself enforces.

This is the short path. The step-by-step reference behind it stays in
[uk-national-release-assembly-runbook-806.md](uk-national-release-assembly-runbook-806.md).

**Status (2026-10-26).** The runner's steps have so far run end to end only
against stand-in tools. Its first real run, by Vahid Ahmadi from his own
laptop with his own registered key, is the last check before it is relied
on for a re-cut; until then the hand-run launchers stay the fallback.

## What you need

| Need | Where it comes from | Check |
|---|---|---|
| A clean checkout of `main` locked at the policyengine-uk you want | a "Lock policyengine-uk X" PR merged to main | `tools/run_uk_cut.py preflight` |
| `HF_TOKEN` | write access to `policyengine/populace-uk-private` and `policyengine/populace-uk-staging`, and read access to the input repos below | preflight |
| `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY` | a key registered **active** in `microcosm.data.gate_signing_keys` (see [Signing keys](#signing-keys)) | `python -m microcosm.data.gate_signing_keys check` |
| Cover under the UK Data Service licences for the FRS, SPI, WAS, LCFS, NTS and ETB | the licence holders | access to the input repositories (below) |
| About 60 GB of free disk, AC power, a quiet host | the machine | preflight |

## One command

```bash
cd <clean checkout of main>
uv sync --all-packages --locked --extra uk
export HF_TOKEN=…
# The key goes to this one command, never to the whole shell:
MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY="$(security find-generic-password -a "$USER" -s microcosm-uk-gate-signing-key -w)" \
  uv run --no-sync python tools/run_uk_cut.py all --label sat24              # build, certify, assemble
# add --publish to publish for inspection
```

The runner passes the key on only to the steps that sign (spine, national
build, certification) or verify (assembly, publication); the dry run, the
scoring, the stage-1 comparison and the preflight tests never see it.

Agent sessions on Max's build Mac read the team key, whose custodian is Max,
with `agent-secret get microcosm/uk-gate-signing-key-team maxghenis`, and
`HF_TOKEN` with `agent-secret get HUGGING_FACE_TOKEN_MAX maxghenis`.

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

- An unfinished spine attempt is left for you to inspect. The builder
  deletes a spine's sidecars when it starts, including a gate-blocked
  attempt's signed report, so a retry needs `--retry-spine`, which moves the
  whole attempt to `spine.failed-<time>` first.
- An unfinished national run is moved aside only with `--retry-national`.
- A score receipt, a certification or a stage-1 comparison that is redone
  keeps the earlier one as `<name>.superseded-<time>`.
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
| `cuts/<label>-<sha9>/cut_receipt.json` | commit, engine, the signing key's fingerprint (from the signed reports), input pins, step timings, tag |

If the machine runs a disk cleaner, keep it away from the work directory for
the length of a run. On a host booked through a reservation tool, pass
`--reservation-tool <path> --reservation-caller <name>`: preflight then asks
`<tool> allowed <name>` before anything is built, and fails if the tool is
missing. Without them no reservation is checked.

### What it will not do

- **It never moves a default.** `--publish` always passes `--no-latest
  --tag-only`, so `populace-uk-private`'s main branch, its root files and
  `latest-*.json` stay where they are.
- **It never promotes.** Promotion (`--promote-line national`) is a separate,
  reviewed call. The runner prints the command for it.
- **It never publishes to the Orrery.** The national step always passes
  `--no-publish-orrery`, as the hand-run launchers do. The graph files stay
  in the run directory; publishing them is a separate, deliberate step
  ([orrery-publication.md](orrery-publication.md)).
- **It never certifies into policyengine.py.** That is a pull request in
  another repository; see [Certifying into policyengine.py](#certifying-into-policyenginepy).
- **It builds only from main.** It refuses a commit that is not on
  `origin/main`, unless you pass `--allow-off-main "<who agreed and why>"` for
  a pushed commit.

### When the build Mac can't: Modal

`tools/modal_uk_cut.py` runs the same command in a Modal container, for when
the Mac is busy, short of disk or asleep. The container:

- clones a pushed commit on main;
- syncs that commit's own lock with the `uk` extra;
- keeps the work directory, with its inputs and every step's outputs, on the
  `microcosm-uk-cuts` volume, so a rerun resumes.

**Processing UKDS microdata on Modal needs the data owners' OK before first
use.** Three things move to Modal with a run:

- `HF_TOKEN` and the signing key, as an ephemeral secret attached to that
  run only. Modal holds both for the run; no named workspace secret keeps
  them. A key used there is then also held on Modal, and its registry
  entry's `held_in` must say so first.
- The UKDS inputs and every output, on the `microcosm-uk-cuts` volume, which
  anyone in the Modal workspace can read. Only a workspace or environment
  restricted to licensed users may hold it.
- The app and its image, built from the pushed commit.

Planning contacts no Modal service and sends nothing. `modal run` creates
the app before its entrypoint can refuse, so it attaches the credentials
only when `MICROCOSM_UK_CUT_MODAL_RUN=1` is set:

```bash
python tools/modal_uk_cut.py --commit <sha on main> --label sat24                         # plan, sends nothing
MICROCOSM_UK_CUT_MODAL_RUN=1 modal run --detach tools/modal_uk_cut.py --commit <sha> --label sat24 --run [--publish]  # paid
```

## Inputs

`tools/uk_cut_inputs.json` says where each input lives. Every expected digest
is read at run time from the pins the build already checks:
`uk/source_stages.json`, `uk/frs_release.json`, `uk/chronicle_feed.json` and
`uk/gates.json`. The one exception is the input-mass reference: the build
binds it by a canonical digest of its contents, so the manifest also carries
the hosted file's byte sha256. A file is installed only if it
is byte-identical to what the build will accept, and nothing unverified is
ever left at an input's path.

**Who can read them.** `policyengine/microcosm-uk-build-inputs`,
`policyengine/populace-uk-private` and `policyengine/populace-uk-staging` are
private. `policyengine/policyengine-uk-data-private`, which holds the UKDS
zips, is a *gated* model repository: anyone can see its file list, and its
files download only after an access request is approved. So that approval
is what admits a builder to the licensed inputs.

| Input | Hosted at | State |
|---|---|---|
| FRS 2024-25 (14 tabs) | `policyengine/policyengine-uk-data-private` `frs_2024_25.zip` @ `a2039519` (gated) | fetched, every member matches its pin |
| SPI 2022-23, WAS round 8, ETB 1977-2024 | the same repo's `spi_2022_23.zip`, `was_2006_22.zip`, `etb_1977_24.zip` | fetched, matches |
| HMRC personal incomes ODS | gov.uk URL pinned in `source_stages.json` | fetched, matches |
| Incumbent enhanced FRS | `policyengine-uk-data-private` `enhanced_frs_2024_25.h5` @ `a9e52499` | fetched, matches |
| Input-mass reference | `policyengine/microcosm-uk-build-inputs` (private) `input-mass-reference/fd41cb5f…/` | fetched, matches |
| Chronicle UK consumer artifact (about 7.4 GB) | `policyengine/microcosm-uk-build-inputs` `chronicle-uk/<the feed's source commit>/`, read at main | fetched, both files match the feed pin. The path follows `chronicle_feed.json`, so each re-pin needs its artifact hosted (below). 272c987's (main's pin) was uploaded on 2026-10-26 from the build #1359 pinned; abf0921's was rebuilt on Modal and reproduced the pin byte for byte |
| NTS 2002-2024 (5 tabs) | `policyengine-uk-data-private` `nts_2002_24.zip` @ `559b69e9` | fetched, every member matches its pin |
| LCFS 2024-25 (2 tabs) | `policyengine-uk-data-private` `lcfs_2024_25.zip` @ `11051052` (a repack of the extracted files) | fetched, every member matches its pin |

### NTS and LCFS

Both were uploaded once by a UKDS licence holder (2026-10-25) and are pinned
by revision and archive sha256 in `tools/uk_cut_inputs.json`:

- `nts_2002_24.zip` is the UK Data Service TAB download of SN 5340 (19th
  edition), unchanged.
- `lcfs_2024_25.zip` (SN 9603, 2nd edition, TAB) is a repack of the
  extracted files: all seven tabs and the two mrdoc workbooks, under
  `lcfs_2024_25/`. The original download zip was no longer available.

The runner finds each tab by name inside the zip and checks it against its
pin, so the folder layout inside does not matter. To check both without
building anything:

```bash
uv run --no-sync python tools/run_uk_cut.py fetch --only nts_2002_2024 --only lcfs_2024_25
```

A tab that differs from its pin is refused by name: that upload is a
different edition from the one the build was reviewed against. A
re-upload needs its new revision and archive sha256 pinned in
`tools/uk_cut_inputs.json`.

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
digests match `chronicle_feed.json`. The 2026-10-22 rebuild reproduced both
(bundle 2h24m, artifact 30 min, peak memory 35.5 GB). Otherwise only the receipt and the bundle
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
  missing key failed only after the solve.

HMAC verification needs the secret that signed. So a cut is signed,
certified and published by one key holder, and the registry holds one key
per builder rather than one shared secret. The runner holds a cut to that:
its preflight refuses to resume a cut under any key but the one its signed
reports name (`attestation.signing_key_sha256`), and `cut_receipt.json`
records that signer, whoever resumed it.

| Label | Fingerprint | Held by | Status |
|---|---|---|---|
| `uk-gate-juaristi22-2026` | `66bf0e29…` | María Juaristi's build machine | active; signed every cut on record |
| `uk-gate-team-2026` | `dd31e325…` | Max Ghenis: agent-secret `microcosm/uk-gate-signing-key-team` on his build Mac, readable by agent sessions on that Mac | active |

**Add a builder or rotate a key.**

1. On macOS, run `uv run --no-sync python -m microcosm.data.gate_signing_keys
   generate --keychain-service microcosm-uk-gate-signing-key`. It stores the
   key in your login keychain through `security -i` on stdin, so the key is
   in no process list, reads it back, and prints only its fingerprint.
   Elsewhere, pipe `generate`'s stdout straight into a secret store: it
   refuses to print a key to a terminal, where scrollback and agent
   transcripts would keep it.
2. Never commit, post or print the key.
3. Open a pull request that adds the fingerprint as `active`.
4. To rotate, set the old entry to `retired` with a `retired_on` date, in
   the same or a later PR. Its cuts still verify, but only for someone who
   holds that key.

**Revoke a leaked key.** Set it to `revoked`. Every cut it signed then fails
verification, so re-sign or withdraw those cuts.

**What the signature does not prove.** The reports are signed with HMAC,
which is symmetric, so:

- Verifying a cut needs the secret that signed it, and whoever can verify
  can also sign. A "certified" cut is its key holder vouching for their own
  reports. policyengine.py and anyone downstream rely on Hugging Face write
  access to the private repositories and on this reviewed registry, not on
  a signature they can check themselves.
- Retiring a key stops the registry's checks from accepting new signatures
  from it, but `retired_on` is only recorded: whoever still holds a retired
  key can mint reports that verify.
- Revocation binds only checkouts and wheels that include the revoking
  commit.

A public-key scheme (Ed25519) would let the registry hold public keys and
anyone verify; that is tracked in microcosm#1406.

## Certifying into policyengine.py

```bash
uv run --no-sync python tools/certify_uk_cut_in_policyengine_py.py \
  --national-tag microcosm-uk-2024-25-national-<ts>-<id> \
  (--dense-tag <tag> | --keep-regional-release | --area-coded-national-uri <uri> | --drop-regional-release) \
  --pepy-repo <your clone of PolicyEngine/policyengine.py> \
  [--max-load1 <load>] [--min-free-gb <GB>] [--push --open-pr]
```

Each heavy step first checks the host. The disk floor defaults to 55 GB
(`--min-free-gb`). There is no load limit unless you set one with
`--max-load1`, because a sensible limit depends on the machine.

For an accepted mismatch (see [Regions](#regions-the-dense-cut)), add
`--accept-regional-model-mismatch "<who accepted it and why>"`,
`--audited-by "<who ran the audit>"` (it appears in policyengine.py's public
bundle) and at least one `--regional-limitation "<text>"`. The base needs
policyengine.py#699's audit, and a base without it refuses before anything
is downloaded. Until #699 merges, test the route with `--base-ref
origin/accept-regional-model-mismatch --pr-base accept-regional-model-mismatch`.
The audit and `certify-data` also need `uv` and PyPI: both read the build
model through a pinned `uv` probe, and `certify-data` re-reads both models
rather than trusting the evidence's lists.

The certification commit and the pull request body credit nobody but the
git identity that commits. An agent session adds its own credit:
`--co-author "Claude Opus 5.5 <noreply@anthropic.com>"` (repeatable) for the
commit trailer, and `--pr-body-footer "🤖 Generated with [Claude Code](https://claude.com/claude-code)"`
for the body.

The script does the mechanical part of what policyengine.py#685 did by hand,
on a new branch from policyengine.py's main.

**Before writing anything:**

- It refuses if another writer is already certifying the cut: an open pull
  request that names the tag, or any open pull request or branch whose name
  contains `certify-uk-<id>`, including suffixed ones such as
  `certify-uk-a133f37a-national`. If it cannot read either listing, it
  refuses rather than passing.
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

policyengine.py routes constituency and local-authority runs only to a dense
cut that covers the certified policyengine-uk. A dense build takes hours, and
its two-build sign-off longer, so a national cut can move to a newer
policyengine-uk before a dense cut on that version exists. There are four
routes:

1. **Ship together.** Build a dense cut on the same version. This is correct
   but slow, and it needs the atomic-support files that only María holds.
   The policyengine-uk 2.151.0 pair (national `a6299504`, dense `c9df4b86`)
   went this way.
2. **Accept the older dense cut, recorded as a mismatch.** policyengine.py
   accepts this only with audited evidence (policyengine.py#699).
   - Area receipts are already uncertified: the certificate covers the
     national dataset only (policyengine.py#685).
   - policyengine-uk loads an older file under a newer model, skipping
     stored columns the model no longer defines. Inputs added since the
     build model default in area runs, and the mismatch record lists them.
3. **Drop routing.** sim-api then cannot serve the UK default's constituency
   and local-authority runs.
4. **An enhanced-FRS release that claims the version.** This needs a new
   uk-data release on Max's go.

**Measured cost of route 2.** This is measured on the 2.132.6 dense file's
own columns (2026-10-23), under policyengine-uk 2.147.2. The UK hub's
rehearsal probe agrees.

- No stored column is dropped on load.
- `pip_dl_category` and `pip_m_category` became formulas in 2.144.0, but the
  file's stored values still load.
- 16 inputs take their defaults in area runs, where the build model computed
  them or did not have them:
  - `salary_sacrifice_returned_to_income` and `local_bus_uncapped_single_fare`;
  - the property flags and debts (`additional_residential_property_purchased`,
    `non_residential_property_purchased`, `non_residential_property_secured_debt`,
    `other_residential_property_secured_debt`);
  - pension inputs (`defined_benefit_pension_share`,
    `pension_annual_allowance_taxable_lump_sum_death_benefits`,
    `pension_contributions_via_salary_sacrifice_pre_2015`);
  - `directly_held_shares`, `unit_and_investment_trusts`,
    `early_years_school_start_deferred`, `is_full_year_category_a_director`
    and `war_widows_pension_reported`;
  - `pip_dl_points` and `pip_m_points`, unused while the stored categories load.
- National figures are unaffected. An area breakdown of a reform that works
  through any of these inputs would be wrong.
- `tools/certify_uk_cut_in_policyengine_py.py` measures this whenever a
  mismatch is accepted, with policyengine.py's own audit:
  - it downloads every regional microdata H5 at its manifest pin and checks
    its sha256;
  - in the certification worktree it installs the target policyengine-uk
    and core, and runs `scripts/audit_regional_input_surface.py`, which reads
    only column names and measures them against the regional cut's build
    model (from its manifest) and the target model;
  - it refuses evidence that is not build-relative (schema 2) or that drops
    a stored column, and passes the evidence to `certify-data`.

  The pull request lists the defaulted inputs, the count of stored columns
  neither model defines (recorded only, such as the area codes) and the
  recorded limitations. There is no override for a dropped column.

Routes 3 and 4 are not recommended. Which route a cut takes is María's
call.
