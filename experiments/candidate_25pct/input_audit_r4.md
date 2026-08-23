# Candidate 25% legacy-arm input audit, round 4

Date: 2026-08-23

Code authority: the release, pool, selection, and scorer sources on this branch
are byte-identical to `origin/main` at `d69131a3534a`. Round-4 branch changes
before this receipt are journals only.

Required label: **one-surface + pkg3, legacy release arm, not exact-k
certified**.

Verdict: **GO for the pool and dense legacy stages. STOP for sparse pending an
owner selection-authority ruling.** No pool, release, selection, or scorer ran
during this audit.

## Cleared full-SCF input

The owner reports that the Federal Reserve SCF 2022 full public Stata dataset
was downloaded from
`https://www.federalreserve.gov/econres/files/scf2022s.zip` and unpacked on
2026-08-23. The installed member was independently inspected and hashed:

```text
path:   /Users/maxghenis/.cache/microcosm/scf/p22i6.dta
size:   236952250 bytes
sha256: 61e2fceb1594e4009eb996d6e25d38a5d8e4874930fc2bfce3c87ffa6946ad0a
file:   Stata Data File (Release 118)
mode:   -rw-r--r--
```

Its opening bytes are the Stata 118 XML header
`<stata_dta><header><release>118</release><byteorder>LSF`; this is not a ZIP,
HTML response, or empty placeholder.

Current main exposes `--scf-full-extract` at
`tools/build_us_fiscal_refresh_release.py:1248-1256`. With the explicit path,
the builder resolves `Path(args.scf_full_extract)` and passes it directly to
`load_scf_2022_auto_loan_donor`
(`tools/build_us_fiscal_refresh_release.py:9579-9587`). The loader reads the
summary/full Stata pair and checks the join and required columns
(`packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py:219-267`).

The default route resolves
`$HOME/.cache/microcosm/scf/p22i6.dta`
(`scf_auto_loans.py:162-180`), which is the same file when the launcher's
explicit `HOME=/Users/maxghenis` is in force. Because current main still has no
archive/member hash constant, a present nonempty cache entry returns without a
hash check (`scf_auto_loans.py:67-77,181-186`). The launcher therefore pins and
hash-checks the owner bytes itself and passes the explicit path.

## Stage 1: immutable pool inputs

These are copied exactly from the incumbent host queue at
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/build-queue.sh:37-53`.
They were independently measured in round 3 and will all be re-resolved and
fully rehashed by the round-4 script's real `--dry-run`.

| CLI input | Host path | Required SHA-256 |
| --- | --- | --- |
| `--asec-raw-stage-h5` | `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5` | `51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe` |
| `--acs-household-zip` | `/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip` | `8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0` |
| `--acs-person-zip` | `/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip` | `afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894` |
| `--acs-rent-h5` | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5` | `0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4` |
| `--puf-h5` | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5` | `7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df` |
| `--puf-source-year-csv` | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv` | `0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df` |

All six path/hash pairs are required by the parser
(`tools/build_us_multispine_pool.py:441-514`). The requested output,
checkpoint, sample, and clone controls exist at lines 515-553. `--out
.../pool/pool.h5` derives `pool.manifest.json` and `pool.gates.json` at lines
580-614.

After a successful pool stage, the launcher must validate the manifest's
simulation-ready status and H5/gates hashes, then persist the exact
`pool.manifest.json` SHA-256. It consumes that wrapper pin again immediately
before dense. It must **not** pass `--pool-manifest` or
`--pool-manifest-sha256` to the legacy builder: either flag enters the exact-k
all-or-none argument group (`tools/build_us_fiscal_refresh_release.py:1500-1516`),
which is incompatible with `--dense-default-dataset` (lines 1568-1571).
Legacy correctly receives the authenticated result through `--base-h5`.

## Stage 2a: dense external inputs

All present external inputs below have fixed paths and measured full-file
SHA-256 values. The round-4 dry-run rechecks each byte stream before printing
the command.

| Purpose | Host path | Required SHA-256 |
| --- | --- | --- |
| Ledger v9.4 facts | `/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl` | `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080` |
| Export input-mass reference | `/Users/maxghenis/PolicyEngine/_buildg-runtime/forensics/populace_us_2024.h5` | `c2065b642ab00da74746afdfd9f06890e5f32f9b10bd6610ff236452d40f39c5` |
| SCF summary extract | `/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/scf_cache/rscfp2022.dta` | `6b8dd2d935a76ed225ddebc80fb2db22a467f0c80d9a1acaa67b4584aa4bafd1` |
| SCF full extract | `/Users/maxghenis/.cache/microcosm/scf/p22i6.dta` | `61e2fceb1594e4009eb996d6e25d38a5d8e4874930fc2bfce3c87ffa6946ad0a` |
| SSI delivered-weight prior basis | `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/buildo-run/dense-p2/releases/populace-us-2024-buildp-dense-cae8640-20260728T050443Z/us_ssi_take_up.json` | `56118bde095b8ef2559a26a3478ff5f8b61939eca402dffcec61189e7de631e3` |
| Official ASEC 2023 archive | `/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education/asecpub23csv.zip` | `d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11` |
| Full SIPP 2023 donor | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023.csv` | `5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2` |
| Slim SIPP tips donor | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023_slim.csv` | `1f0bcb8e045ef1118e8eba4b4a2997bdaaf947bd0dd09d41fa7c7d5657a3d7d5` |
| CPS ORG wages donor (compressed file) | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/census_cps_org_2024_wages.csv.gz` | `66fa5b6aa4087413b691038767b51f603281ff55411b58259922f78e67460372` |
| Packaged CD vintage crosswalk | `packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv` | `c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec` |

The explicit donor flags exist at
`tools/build_us_fiscal_refresh_release.py:1227-1285`; supplying them prevents a
launch-time network fetch. The CD crosswalk is an unconditional packaged
default at lines 1383-1393 and 1472-1478. The ORG loader additionally enforces
the canonical decompressed-content pin
`d74600236cfdd34033d487cd9d82f6eb00b1858ba28de17f4f985f2aec516f86`
(`packages/microcosm-build/src/microcosm/build/us_runtime/org_wages.py:81-88,500-510`).

The July dense invocation is the epoch/control authority
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp2_dense.sh:63-80`):
`--seed 0`, `--epochs 3000`, `--dense-default-dataset`,
`--skip-reform-validation`, and `--no-staging`. Every flag remains present at
current HEAD (`tools/build_us_fiscal_refresh_release.py:1066-1075,1130-1138,
1226,1298-1313,1368-1372,1448-1451`). The Ledger, export-reference, and
release output flags remain at lines 831-935 and 976-987.

The incumbent SSI basis still matches current main. It is schema 4,
`release_final`, and was measured on a `current_frame` basis, so it is not a
forbidden retry-of-retry. Its three SSA target values exactly match v9.4. The
current loader still requires the path/hash pair and validates hash, schema,
phase, target contract, capacity, and gate integrity
(`tools/build_us_fiscal_refresh_release.py:1150-1177,5960-6028` and
`packages/microcosm-build/src/microcosm/build/us_runtime/ssi_take_up.py:902-1048`).
It is therefore the incumbent-modeled basis for dense.

The only operator-provided reviewed-exclusion register on the current parser is
`--qrf-tail-concentration-exclusions` (lines 909-918). It is omitted. Omission
loads `{}` and is recorded with a null path/hash and empty mapping
(`tools/build_us_fiscal_refresh_release.py:7790-7803,10934-10966`). The retired
`--zero-support-exclusions` flag must also remain absent; current tests require
its rejection
(`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:71-93`).
This implements zero operator waivers without pretending that current main's
checked-in source authorities are replaceable by CLI registers.

## Stage 2b: sparse selection-authority STOP

Current main cannot derive an exact sparse-57k authority from a full new pool.
`tools/build_us_selection_source_manifest.py` is only a wrapper around the
selection manifest distiller. The implementation extracts **every** household
identity from an already-selected H5 and writes those identities
(`packages/microcosm-build/src/microcosm/build/us_runtime/warm_start_selection.py:371-413,477-505`).
Its CLI has source, output, join-key, and provenance arguments only; it has no
record count, seed, filter, pool-selection, or L0 argument (lines 509-539).
Passing the new pool H5 would therefore freeze the whole pool, not 57,240
records.

The legacy cold L0 route is not an exact-count selector. It uses the default
fixed penalty share `0.8` (`tools/build_us_fiscal_refresh_release.py:1093-1103,
10132-10140`) and records whatever count results (lines 10326-10367). The
generation contract explicitly sets `target_records` to null
(`tools/us_bundle_generation/contracts.py:1414-1427`). Exact 57,240 is available
only through the separately ratified exact-k arm, which is outside this
owner-ruled legacy candidate and is mutually exclusive with frozen selection
and dense polish.

The incumbent sparse script does establish what a future authorized frozen
support invocation would keep: selection manifest plus
`--dense-default-dataset`, `--selection-mass-protection
keogh_distributions`, and `--epochs 6000`
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp_sparse9.sh:123-143`).
Frozen support followed by dense polish is the documented construction
(`docs/informed-l0-warm-start-design.md:137-155`). Keogh protection remains
current doctrine: the parser says it protects a selected carrier mass until a
real Ledger fact exists (`tools/build_us_fiscal_refresh_release.py:921-932`),
v9.4 contains no `keogh_distributions` fact, and the current target is appended
after parity at lines 10032-10045. It is a synthetic calibration target, not a
waiver.

The launcher must therefore end dense-only with this exact question:

> **Owner ruling required:** which rule may choose the candidate pool's
> support: (A) current legacy fixed-penalty L0 at default 0.8, accepting its
> non-exact realized count, or (B) a newly ratified exact-57,240 rule, including
> its algorithm, seed, and Keogh-carrier treatment? Current main has no legacy
> CLI that derives an exact 57,240 selection manifest from a pool.

No selection path, pin, identity list, sparse release ID, or sparse artifact is
invented before that ruling.

## Scorer inputs

The committed incumbent evidence
`experiments/replacement_scorecard/incumbent_48b9d479.json` has SHA-256
`b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8`.
It maps to the host incumbent H5:

```text
/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5
sha256 48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e
```

The scorer requires `--incumbent`, optional `--candidate`, `--ledger-facts`,
and `--out-prefix` (`tools/score_us_release_head_to_head.py:260-311`). The JSON
is evidence, not the `--incumbent`: a `.json` argument is parsed as a pool
manifest (lines 495-513). The launcher will print the dense H5 command and will
print a sparse command only if a sparse artifact is actually produced.

## Runtime guards selected from evidence

- Pool: 90 GiB reclaimable-memory gate; same-week f025 attempts took about
  2.8 hours and peaked at 75.46 and 85.85 GiB.
- Dense: 110 GiB reclaimable-memory gate; the July attempt took 1 h 39 m 07 s,
  failed, and its pressure log peaked at 99,149 MiB (96.83 GiB).
- Sparse, if later authorized: 90 GiB; the successful July run took 2 h 42 m
  03 s and `/usr/bin/time -l` recorded 79.17 GiB maximum RSS.

“Reclaimable” is memory from `vm_stat` free + inactive + speculative +
purgeable pages, not filesystem capacity. The thresholds are scheduling
preconditions, not predictions or tuning parameters. Exact sources are carried
into `_LANE-NOTES.md`.
