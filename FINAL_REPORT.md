# Final report: 25% candidate runbook, round 4

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

Required label: **one-surface + pkg3, legacy release arm, not exact-k
certified**.

Outcome: **the dense-first launcher, complete input audit, real exit-0 dry-run,
and lane handoff are committed. Sparse correctly stops for an owner ruling. No
pool or release build ran.** The only incomplete delivery is the requested
external launcher copy: the managed filesystem denied writes to
`/Users/maxghenis/PolicyEngine/_buildo-runtime`, so the executable exists only
at the committed canonical path until the owner performs the exact-byte copy
below.

## Cleared blocker and input authority

The owner-supplied full Federal Reserve SCF 2022 extract exists at
`/Users/maxghenis/.cache/microcosm/scf/p22i6.dta`, is 236,952,250 bytes, and
hashes to
`61e2fceb1594e4009eb996d6e25d38a5d8e4874930fc2bfce3c87ffa6946ad0a`.
Its header identifies Stata release 118 with little-endian `LSF` byte order.
Current main's explicit `--scf-full-extract` route resolves that exact path and
passes it to the Stata loader
(`tools/build_us_fiscal_refresh_release.py:1248-1256,9579-9587`). The source
URL, unpack date, header bytes, loader trace, and hash are recorded in
`experiments/candidate_25pct/input_audit_r4.md`.

The dry-run fully rehashed all 18 immutable inputs:

- the six pool inputs copied exactly from the incumbent host queue;
- Ledger v9.4 at the required
  `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`
  pin;
- the July export-mass reference, SSI prior-weight basis, SCF summary/full
  extracts, ASEC archive, SIPP donors, ORG donor, and packaged CD crosswalk;
- the committed incumbent evidence JSON and the actual 462,915,783-byte
  incumbent H5 at SHA-256
  `48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e`.

Round 3's surface finding stands: the owner-ruled legacy arm compiles the same
unified fiscal target registry as exact-k. Dense and sparse remain separate
stage-2 invocations.

## Launcher delivered in the repository

The executable source is
`experiments/candidate_25pct/run-candidate.sh` (mode `0755`), SHA-256:

```text
94f113bf1d3d7fc58c3973549b66a8aef9f2a3d55931c5f0ea60560e65f16a1a
```

It provides:

- explicit launchd-safe `PATH`, `HOME=/Users/maxghenis`, and
  `PYTHONUNBUFFERED=1`;
- unconditional off-chain execution by unsetting
  `POPULACE_LOGBOOK_PREV_ROW_DIGEST`, wrapping builders with `env -u`, omitting
  `--logbook-prev-row-digest`, and never reading or writing pending-chain state;
- serial pool then dense execution, no publication/promotion path, append-only
  main/stage logs, and 30-second process-tree RSS CSV sampling;
- authenticated idempotent skips, including exact pool sampling/clone receipts,
  all six pool provenance pins, H5/gates hashes, the pool-manifest wrapper pin,
  the dense base-dataset link, code/Ledger identity, dense method/epochs, empty
  reviewed exclusions, enforced/passing QRF tails, and disabled staging;
- checkpoint resume after a pool gate-failed diagnostic trio rather than
  incorrectly accepting it as complete;
- a persisted root commit pin and dense release ID, with clean-HEAD and
  byte-identical launcher rechecks before launch;
- pre-launch authentication/readiness loops. Every real builder waits for no
  running pool/release builder, the stage's reclaimable-memory threshold, AC
  power, and the `.max-go` marker, polling every 300 seconds. If readiness
  changes during hashing, authentication repeats before launch;
- `com.microcosm.candidate25` self-removal on actual-mode exit.

Stage 1 renders the required f025 command with sample fraction `0.25`, sample
seed `578`, full clone attachment, clone seed `578`, the six exact queue
inputs, `candidate-25/pool/checkpoints`, and `candidate-25/pool/pool.h5`. A
successful simulation-ready manifest is hashed into
`pool.manifest.sha256`; dense rehashes and consumes it through the legacy
`--base-h5` route. It is deliberately not passed as `--pool-manifest`, because
that current parser group is exact-k-only and incompatible with
`--dense-default-dataset`.

Stage 2a renders one dense legacy release with `--dense-default-dataset`, seed
0, 3,000 epochs, Ledger v9.4 and its pin, the verified July SSI basis and pin,
the reference H5, both SCF extracts, all other explicit donors,
`--skip-reform-validation`, `--no-staging`, and no reviewed-exclusion register.
The release ID is
`populace-us-2024-onesurface-pkg3-legacy-dense-<sha8>-<UTC timestamp>`.

At completion the launcher prints the dense artifact path and SHA-256, the
sparse non-production status, the incumbent evidence identity, and the exact
`tools/score_us_release_head_to_head.py` dense command using the incumbent H5,
candidate H5, Ledger JSONL, and output prefix. It never runs the scorer itself.

## Sparse stage: intentional STOP

Current main cannot derive an exact new 57,240-household selection authority
from the candidate pool. Its selection-manifest tool records every identity
from an already selected H5; it has no count, seed, filtering, or L0 selection
interface. Passing the full pool would freeze the whole pool. The legacy cold
L0 route instead uses a fixed penalty of 0.8 and records whatever non-exact
count results. Exact 57,240 is available only through the separately ratified
exact-k arm, outside this legacy ruling and missing the owner's `pi_hi` and
artifact pins.

The launcher therefore prints a dense-only STOP and this exact decision:

> Which rule may choose the candidate pool support: (A) current legacy
> fixed-penalty L0 at default 0.8, accepting its non-exact realized count, or
> (B) a newly ratified exact-57,240 rule, including algorithm, seed, and
> Keogh-carrier inclusion policy? If B uses current exact-k, supply `pi_hi` and
> its artifact pins.

The incumbent evidence still establishes 6,000 epochs and
`--selection-mass-protection keogh_distributions` for any future authorized
frozen-support invocation. Current doctrine still requires Keogh carrier
protection because Ledger v9.4 has no real Keogh fact. Neither fact authorizes
reuse of the incumbent's old-pool selection manifest.

## Real dry-run and verification

The committed canonical launcher was executed, not simulated:

```text
./experiments/candidate_25pct/run-candidate.sh --dry-run
exit 0
```

It ran from commit `6327ec0208803a811f59783723a2cd4df5824ad2` and took
29 seconds. Every data hash and SCF header check passed; every used current
parser flag passed; the complete pool, dense, and scorer command lines printed;
sparse stopped at the owner question; and the terminal receipt confirmed no
builder, selection, release, scorer, publication, promotion, staging, or
launchd mutation. The candidate output root remained absent. The full stdout
is committed verbatim in `experiments/candidate_25pct/dry_run_r4.md`.

Validation completed:

```text
bash -n experiments/candidate_25pct/run-candidate.sh       PASS
shellcheck experiments/candidate_25pct/run-candidate.sh    PASS
.venv/bin/python tools/score_us_release_head_to_head.py --help
                                                            PASS
git diff --check                                           PASS
```

The live scorer help confirms that `--incumbent`, `--ledger-facts`, and
`--out-prefix` are required and `--candidate` accepts an entity H5; the printed
command uses exactly that surface.

No pool/release build, publication, promotion, push, tuning, or scorer run
occurred. No Python test suite was warranted for the shell-and-journal-only
changes. No pending Logbook chain state was touched.

## Runtime handoff

The evidence-backed dense-only planning window is about 4.5 measured hours;
reserve 6 hours plus any readiness wait:

- pool: about 2.8--3.0 h, measured 70.34--85.85 GiB maximum RSS, 90 GiB
  reclaimable-memory gate;
- dense: nearest July analogue 1 h 39 m 07 s, measured 96.83 GiB peak, 110 GiB
  gate; the exact new-pool wall/RSS remains unmeasured;
- sparse: not scheduled. Conditional July evidence is 2 h 42 m 04 s and 79.17
  GiB, with a future 90 GiB gate if the owner authorizes selection.

Exact log/status locations and field/line citations are in `_LANE-NOTES.md`.

## Managed-filesystem blocker and owner action

The required external directory did not exist. This authorized install attempt
failed at the sandbox boundary:

```text
mkdir: /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25: Operation not permitted
```

No external file is claimed. The owner must first install the committed exact
bytes:

```bash
mkdir -p /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25 && /usr/bin/install -m 755 /Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/experiments/candidate_25pct/run-candidate.sh /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/run-candidate.sh
```

Then the requested one-line launch action is:

```bash
launchctl submit -l com.microcosm.candidate25 -- /Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/run-candidate.sh
```

The worktree must be clean when launched. The launcher will pin the then-current
full commit and embed its SHA-8 in the persisted dense release ID.

## Round-4 commits before this report

- `c02a4d56` — start the round-4 progress journal.
- `28a179e2` — clear dense inputs and stop unruled sparse selection.
- `430b341a` — add the guarded dense-first candidate launcher.
- `55eb2452` — close the readiness/authentication race.
- `a695afa4` — require the enforced QRF tail gate on dense skips.
- `6327ec02` — report dereferenced input sizes.
- `98da000b` — record the round-4 real dry-run.
- `40f25a1f` — document the candidate lane runtime handoff.

No push was made.
