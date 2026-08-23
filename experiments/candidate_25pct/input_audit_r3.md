# Candidate 25% legacy-arm audit, round 3

Date: 2026-08-23

Code authority: `d69131a3534a` (`origin/main` at round-3 start). The release
builder and its package sources in this branch are byte-identical to that
reference; branch-only differences before this receipt are journals.

Required label: **one-surface + pkg3, legacy release arm, not exact-k
certified**.

Status: **STOPPED at the mandatory full-SCF input.** The target-surface gate
passed, but the current-main legacy loader requires a full 2022 SCF extract
that is absent on this host and has no source pin in code. No pool or release
builder has run.

## 1. Unified target-surface verification

Verdict: **GO on target-surface identity.** The owner-ruled legacy dense arm
does not compile a different or smaller fiscal target surface than exact-k.

PR #741 is merge `c31e1525f099` ("Compile one target surface unconditionally;
remove membership flags"). Current tests require the parser to reject all three
former per-run membership controls:
`--include-congressional-district-targets`,
`--diagnostic-skip-tax-expenditure-targets`, and
`--zero-support-exclusions`
(`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:52-93`).
The current parser unconditionally resolves the packaged congressional-district
crosswalk for every run
(`tools/build_us_fiscal_refresh_release.py:1472-1478`).

Exact-k and legacy initially differ in base identity handling: exact-k
authenticates a pool manifest/H5, whereas legacy takes a bare `--base-h5` and
hashes it (`tools/build_us_fiscal_refresh_release.py:8213-8258`). Both paths
then reach the one unconditional Ledger loader and the same single call to
`compile_us_fiscal_target_registry`
(`tools/build_us_fiscal_refresh_release.py:8358-8371`). The same Medicaid
substitutions and target-parity gate follow before either calibration arm
branches (`tools/build_us_fiscal_refresh_release.py:8372-8402`).

The compiler signature has facts, period, crosswalk, aging, and period-waiver
inputs only; it has no exact-k, dense, geography-membership, JCT-skip, or
per-run exclusion argument
(`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-940`).
It builds the shared dynamic-fact and JCT-reference registry and applies the
same transforms, aging, and period contract
(`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:962-1017`).
The checked-in parity authority states the one-surface rule directly: national,
state, and congressional-district facts compile for every artifact, while
record count changes only selection, never target membership
(`packages/microcosm-build/src/microcosm/build/us/target_parity_manifest.json:524-526`).

After the common source stages, both arms pass the same `target_specs` through
one `_load_or_materialize_target_frame` call
(`tools/build_us_fiscal_refresh_release.py:10070-10108`). Only then does exact-k
pass `registry.to_target_set()` to `calibrate_exact_k_ladder`
(`tools/build_us_fiscal_refresh_release.py:10207-10246`) while legacy dense
passes that same target set to `calibrate`
(`tools/build_us_fiscal_refresh_release.py:10300-10316`). A bare Ledger JSONL
and an artifact directory therefore compile the same facts bytes identically;
the envelope changes provenance and exact-k eligibility, not compiler
membership
(`packages/microcosm-build/src/microcosm/build/ledger_artifact.py:40-56,88-99,132-165`).

`--dense-default-dataset` itself:

- relaxes the L0-lambda validation and forbids a refit-L2 override
  (`tools/build_us_fiscal_refresh_release.py:1479-1490`);
- is mutually exclusive with exact-k
  (`tools/build_us_fiscal_refresh_release.py:1568-1571`);
- selects full-pool calibration and records `dense_no_l0` identity metadata
  (`tools/build_us_fiscal_refresh_release.py:10132-10140,10300-10325`); and
- activates the dense-arm SSI delivery enforcement fences after calibration
  (`tools/build_us_fiscal_refresh_release.py:10448-10462`).

Thus the stronger claim that the flag changes *only*
selection/identity/publication handling needs one explicit qualification: it
also changes post-calibration SSI release-gate enforcement. That does not
change fiscal target membership, so it does not trigger the owner's
different-surface stop rule, but it must remain in the release-arm label and
evidence.

Two controls must stay absent. `--selection-mass-protection` can append
synthetic run-scoped calibration targets after the parity gate
(`tools/build_us_fiscal_refresh_release.py:921-932,10032-10045`), and is not in
the owner ruling. `--gate-congressional-district-targets` changes only whether
unmaterializable CD rows are hard failures, not which materializable specs
enter the registry (`tools/build_us_fiscal_refresh_release.py:4323-4347,
6451-6474`). Materializability still depends on the actual frame, so a finished
legacy run must record and compare its `target_surface` count and SHA-256; the
code path itself has no legacy-only narrowing.

## 2. Ordered stop

The first unrecoverable stage-2 input is the full 2022 SCF extract. The July
launcher did not pass `--scf-full-extract`, so current main resolves the
default path:

```text
/Users/maxghenis/.cache/microcosm/scf/p22i6.dta
```

That file is absent. The adjacent source archive
`/Users/maxghenis/.cache/microcosm/scf/scf2022s.zip` is also absent. Scoped
searches of the relevant PolicyEngine runtimes, the canonical cache, and the
host file index found no other `p22i6.dta` or `scf2022s.zip`.

The parser exposes this input at
`tools/build_us_fiscal_refresh_release.py:1248-1256`; the release flow resolves
and reads it unconditionally at
`tools/build_us_fiscal_refresh_release.py:9572-9587`. The loader's default
download path is implemented at
`packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py:162-216`,
but both the archive and member SHA-256 constants are `None`
(`packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py:67-77`).
Downloading would therefore mint an unreviewed input rather than resolve a
pinned one. Per the owner's stop rule, it was not attempted, no substitute was
used, and no SHA-256 can honestly be recorded for the missing file.

Two independent contract conflicts reinforce the stop:

1. One `--dense-default-dataset` invocation enters only the `dense_no_l0`
   calibration branch and records `sparse=False`
   (`tools/build_us_fiscal_refresh_release.py:10300-10325`). It writes one H5,
   `<out>/artifacts/populace_us_2024.h5`
   (`tools/build_us_fiscal_refresh_release.py:11060-11066`). The sparse L0/refit
   artifact requires the separate non-dense branch
   (`tools/build_us_fiscal_refresh_release.py:10326-10367`). A single
   owner-ruled legacy-dense release cannot truthfully print and score both a
   dense and a sparse artifact.
2. Current receipts support the requested roughly 2.8-hour pool duration, but
   not the proposed roughly 22 GiB peak. The two f025 receipts record 2.819 h
   and 75.46 GiB, and 2.799 h and 85.85 GiB, respectively; both runs exited
   nonzero. Exact sources are recorded in section 7.

## 3. July explicit stage-2 inputs and controls

The starting authority is the incumbent invocation at
`/Users/maxghenis/PolicyEngine/_buildo-runtime/scripts/buildp2_dense.sh:63-80`.
All hashes below were measured from the named host bytes with SHA-256.

### Base pool H5

July path:

```text
/Users/maxghenis/PolicyEngine/_buildo-runtime/out/base-p2/base_populace_us_2024_puf_support.h5
```

- Size: `2228722025` bytes.
- Measured SHA-256:
  `fca79b422cb3c41eb3a7ecfa5906ff012a8e8d21348748cf77ece4ad0667924c`.
- Requirement: `--base-h5` parser at
  `tools/build_us_fiscal_refresh_release.py:831-835`; legacy selection and
  hashing at `tools/build_us_fiscal_refresh_release.py:8256-8259`; file hash
  helper at `tools/build_us_fiscal_refresh_release.py:1626-1629`; H5 load at
  `tools/build_us_fiscal_refresh_release.py:2454-2471`.

For the candidate, this input must instead be the as-built output of its stage-1
25% pool. It does not yet exist and therefore has no path-derived release ID or
SHA-256 to record. The July H5 is evidence only and was not substituted.

### Ledger facts

```text
/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl
```

- Size: `131852600` bytes.
- Measured and owner-pinned SHA-256:
  `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`.
- Requirement: `--ledger-facts` and `--ledger-facts-sha256` parser at
  `tools/build_us_fiscal_refresh_release.py:871-896`; unconditional load before
  target compilation at `tools/build_us_fiscal_refresh_release.py:8358-8371`;
  bare-file verification at
  `packages/microcosm-build/src/microcosm/build/ledger_artifact.py:88-166`.
  A manifest pin is required only for exact-k
  (`tools/build_us_fiscal_refresh_release.py:1563-1566`).

### Export input-mass reference

```text
/Users/maxghenis/PolicyEngine/_buildg-runtime/forensics/populace_us_2024.h5
```

- Size: `354374956` bytes.
- Measured SHA-256:
  `c2065b642ab00da74746afdfd9f06890e5f32f9b10bd6610ff236452d40f39c5`.
- Requirement: parser at `tools/build_us_fiscal_refresh_release.py:976-987`;
  load and comparison at `tools/build_us_fiscal_refresh_release.py:10836-10869`;
  H5 mass loader at
  `packages/microcosm-build/src/microcosm/build/us_runtime/l0_refit_export.py:485-504`.

### SCF summary extract

```text
/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/scf_cache/rscfp2022.dta
```

- Size: `24904185` bytes.
- Measured SHA-256:
  `6b8dd2d935a76ed225ddebc80fb2db22a467f0c80d9a1acaa67b4584aa4bafd1`.
- Requirement: parser at `tools/build_us_fiscal_refresh_release.py:1237-1246`;
  resolution at `tools/build_us_fiscal_refresh_release.py:9377-9395`; pinned
  loader at
  `packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py:154-173,560-608`.

### SSI take-up prior-weight basis

```text
/Users/maxghenis/PolicyEngine/_buildo-runtime/out/buildo-run/dense-p2/releases/populace-us-2024-buildp-dense-cae8640-20260728T050443Z/us_ssi_take_up.json
```

- Size: `3957` bytes.
- Measured and July-pinned SHA-256:
  `56118bde095b8ef2559a26a3478ff5f8b61939eca402dffcec61189e7de631e3`.
- Requirement: path/hash parser contract at
  `tools/build_us_fiscal_refresh_release.py:1151-1177`; paired resolution at
  `tools/build_us_fiscal_refresh_release.py:8407-8412`; content validation at
  `tools/build_us_fiscal_refresh_release.py:5960-6028`.

### Reviewed QRF exclusions: omit

The July invocation used:

```text
/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/qrf_tail_exclusions_densep2.json
```

- Size: `6577` bytes.
- Measured SHA-256:
  `d1484b163748ed360fe51cf7b2a5e23e4514d920bdaf997f31e696c4b2750339`.
- It contains nine reviewed waivers and is **not authorized** for this
  candidate. The flag is optional (`tools/build_us_fiscal_refresh_release.py:909-918`)
  and the loader maps an omitted path to the empty register `{}`
  (`tools/build_us_fiscal_refresh_release.py:7790-7803`), consumed at
  `tools/build_us_fiscal_refresh_release.py:10934-10966`.

The zero-waiver implementation is therefore to omit
`--qrf-tail-concentration-exclusions`; no empty placeholder file is needed.

### Runtime controls that are not immutable input files

- `--checkpoint-root` is mutable resume/output state, not a reviewed source.
  Its parser and resolver are at
  `tools/build_us_fiscal_refresh_release.py:1298-1348,2391-2419`, and cache
  identity validation is at
  `tools/build_us_fiscal_refresh_release.py:4350-4399`. A candidate run would
  require a fresh root under its release workspace; the July checkpoint must
  not be reused.
- `--seed 0` is the incumbent value and the legacy default
  (`tools/build_us_fiscal_refresh_release.py:1226,1581-1583`).
- The exact July epoch count is `--epochs 3000`; parsing is at
  `tools/build_us_fiscal_refresh_release.py:1066-1075`.
- `--skip-reform-validation` is parsed/used at
  `tools/build_us_fiscal_refresh_release.py:1368-1372,11144`.
- `--no-staging` is parsed/used at
  `tools/build_us_fiscal_refresh_release.py:1448-1451,8079-8080`.

## 4. Current-main implicit/default stage-2 files

These files are reached by the legacy base-H5 flow even though the July
launcher did not name all of them.

### Congressional-district vintage crosswalk

```text
/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv
```

- Size: `77935` bytes.
- Measured SHA-256:
  `c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec`.
- Requirement/default: `tools/build_us_fiscal_refresh_release.py:1383-1393,1472-1478`;
  load/hash at `tools/build_us_fiscal_refresh_release.py:8284-8297`; packaged
  resolution at
  `packages/microcosm-build/src/microcosm/build/us_runtime/congressional_district_vintage.py:85-94`.

### ASEC official archive

```text
/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education/asecpub23csv.zip
```

- Size: `150165063` bytes.
- Measured SHA-256:
  `d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11`.
- Requirement: parser at `tools/build_us_fiscal_refresh_release.py:1227-1235`;
  legacy base-H5 flow resolution/load at
  `tools/build_us_fiscal_refresh_release.py:8470-8488`; ZIP/member verification
  at
  `packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py:473-520,560-587`.

### Full SIPP 2023 donor

```text
/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023.csv
```

- Size: `3726010471` bytes.
- Measured SHA-256:
  `5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2`.
- Requirement: historical `--sipp-vehicle-donor` parser at
  `tools/build_us_fiscal_refresh_release.py:1266-1275`; shared resolution at
  `tools/build_us_fiscal_refresh_release.py:9382-9395`; financial-assets and
  vehicle-donor loads at
  `tools/build_us_fiscal_refresh_release.py:9432-9475,9619-9660`; exact-path,
  size, and pin checks at
  `packages/microcosm-build/src/microcosm/build/us_runtime/sipp_financial_assets.py:241-304`.

### Full SCF 2022 extract — missing, ordered stop

```text
/Users/maxghenis/.cache/microcosm/scf/p22i6.dta
```

- Status: **absent**; no measured SHA-256 exists.
- Requirement and loader: section 2 above.

### Slim SIPP tips donor

The default cache path is absent, but the loader's exact pinned
`policyengine-us-data` storage path is present and may be supplied explicitly:

```text
/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023_slim.csv
```

- Size: `65649523` bytes.
- Measured SHA-256:
  `1f0bcb8e045ef1118e8eba4b4a2997bdaaf947bd0dd09d41fa7c7d5657a3d7d5`.
- Requirement: parser at `tools/build_us_fiscal_refresh_release.py:1257-1265`;
  runtime resolution at `tools/build_us_fiscal_refresh_release.py:9692-9700`;
  pin and load at
  `packages/microcosm-build/src/microcosm/build/us_runtime/sipp_tips.py:70-83,284-299`.

### CPS ORG wages donor

The default cache path is absent, but the loader's exact pinned
`policyengine-us-data` storage path is present and may be supplied explicitly:

```text
/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/census_cps_org_2024_wages.csv.gz
```

- Size: `1677834` bytes.
- Measured compressed-file SHA-256:
  `66fa5b6aa4087413b691038767b51f603281ff55411b58259922f78e67460372`.
- Measured canonical decompressed-content SHA-256:
  `d74600236cfdd34033d487cd9d82f6eb00b1858ba28de17f4f985f2aec516f86`.
- Requirement: parser at `tools/build_us_fiscal_refresh_release.py:1276-1285`;
  runtime resolution at `tools/build_us_fiscal_refresh_release.py:9733-9741`;
  content pin and verification at
  `packages/microcosm-build/src/microcosm/build/us_runtime/org_wages.py:81-88,500-510`.

### Packaged parity authorities

```text
/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/packages/microcosm-build/src/microcosm/build/us/ecps_parity_reference.json
/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/packages/microcosm-build/src/microcosm/build/us/ecps_parity_known_gaps.json
```

- Reference measured SHA-256:
  `9cad4085d31701371f60e65b415fb03978fd7f4df328292647c08f81aa49eab0`.
- Known-gaps measured SHA-256:
  `6d5cddc37bd9d1a0a4270f90188e15bf7c7793960f8717d2c583949bd8650bb3`.
- Loaders:
  `packages/microcosm-build/src/microcosm/build/us_runtime/parity_reference.py:118-158,225-258`.

These are code-owned authorities fixed by the source commit, not mutable
operator exclusion registers. The checked-in target parity manifest and
selection policies are likewise part of the code pin; no run-scoped
membership or mass-protection override is authorized.

### Remaining code-owned authorities

The external/operator file list above is complete. Static inspection also
found the following package-data inputs. They travel with code authority
`d69131a3534a`; they are not paths or pins the launcher may replace. Paths in
this table are relative to the exact host root
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/packages/microcosm-build/src/microcosm/build/`.

| Path below package root | Measured SHA-256 | Requiring loader |
| --- | --- | --- |
| `us/source_stages.json` | `a3e9ca87f43d74b3d83320ca77559f28452036cf60dfc16bee10a22d4784f672` | `us_runtime/__init__.py:2460-2475`; release preflight also reads it at `us_runtime/validation_input_coverage.py:175-190` |
| `us/support_spine.json` | `68f37dc6ae6e0cde7ebccb53f88dd4a800e63456f838fa214ff98d1db8d815be` | import-time load at `us_runtime/__init__.py:2466-2475` |
| `us/fiscal_target_references.json` | `46a29d809596a46114e0b1e8362330b70c2fadf4f51c49844f1b93b5fb776781` | target compiler loader at `us_runtime/fiscal_targets.py:922-930` |
| `us/release_input_coverage_manifest.json` | `270251ef9d043ed8fb3a0c2e19ddbbf0fe9b9edbb59922f706a3b99085aa5712` | `us_runtime/release_input_coverage.py:409-434`, called by the release preflight at `tools/build_us_fiscal_refresh_release.py:8321-8326` |
| `us/target_parity_manifest.json` | `da2247f94c4a9ad73936a8b225f2ede838220383f417e9f9c741820f6cfb9ee1` | `us_runtime/release_target_parity.py:349-374`, called at `tools/build_us_fiscal_refresh_release.py:8384-8392` |
| `us/target_parity_feed_families.json` | `ab583523e5ba0fdbf1196d2c40a1b965f647fed2b083183f7676f5fd301dec34` | feed inventory load/assertion at `us_runtime/release_target_parity.py:427-443,519-550` |
| `us/puf_aggregate_record_disaggregation.json` | `4e6d2f4f8fc2c1b708a8450d75ca5693c0acf74138a1a6b6779652949f2f32b1` | default spec loader at `us_runtime/puf_aggregate_records.py:226-232`; candidate-tail checks call that default through `us_runtime/puf_capital_gains_tail.py:192,249,312` |
| `us/soi_table_2_1_interest_components_ty2015.json` | `c3356ae216487f365cb0e0a7ab1ba46843a6c52950b75eae1bfab9b0b80a735a` | import-time source-asset load at `us_runtime/puf_interest_components.py:100-151,202-205` |
| `us/soca_capital_gain_distribution_shares.json` | `e7d31a5956dc420940002b5c5120b4fa7f6af6fa8bf071d488affe35b616611d` | share loader at `us_runtime/capital_gain_distributions.py:120-140,227` |

The typed take-up preflight additionally reads the following frozen package
surface. `load_country_take_up_contract_projection` selects these resources at
`country_spec.py:1310-1409`; the ABI lock is read and checked at
`spec_engine/engine_abi.py:410-496`.

| Path below package root | Measured SHA-256 |
| --- | --- |
| `us/country_package.json` | `39840b9245f635ca1dac99102faf76da44e9b06ef5a01abd429152b95273d87c` |
| `us/spec/sources.yaml` | `cb2a34d98ec5fc1b3bb1a8358af54c236e4af142eb7c060c339f6d4c537f7880` |
| `us/spec/take_up.yaml` | `bafba1a0cf18b1d360fd7b6623c9fdc55893e167dc6a08ad90e4912cf6bdcd5d` |
| `us/take_up_contract.json` | `5852e96582793313782d1c3edfc4cfdd0358a1a9cfd54bfea5844cbb09e89bd4` |
| `us/engine_abi.lock.json` | `cc1a71f38d0e40a3acb9f199edb5da91db6397ee4a5b50fe0fcac89a79c981ea` |

That narrow projection initializes the shared schema registry, which reads all
15 JSON schema files at `spec_engine/schemas.py:445-493`. Paths below are
relative to the exact host root
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook/packages/microcosm-build/src/microcosm/build/spec_engine/schema/`.

| Schema file | Measured SHA-256 |
| --- | --- |
| `battery.schema.json` | `68be3b8d9ec206e5306e10ccd49a92dd3a5fb86958780c3561d1caa28ee7a51c` |
| `bundle.schema.json` | `96d35e7b0831653fc7a4966914cecd8ce99ae3a7577b7cff2262378294c05add` |
| `calibration.schema.json` | `719a73266cc86e7019cfab91f3272c1db1316a0f70e060c8dcfbeb7af5151e32` |
| `catalogs.schema.json` | `472c86eb28d66dc571e5968d4407245740bd3dc605b608e73adfb8cd048322fa` |
| `defs.schema.json` | `73f8e40690a3bdfa38e6f8e3d370d6483a14336ea1d47f4675fefbdf9239c60d` |
| `geography.schema.json` | `c74a2e4e60ea179a78989408e4e702d453d5aa38765d6159db57999c719f04b1` |
| `imputation.schema.json` | `71a3a017412d6095fa9f03c187408f14b4c38b5cae33eaa2e4da62767c7a7d62` |
| `locks.schema.json` | `d4f7a8a960fb637d416f3e31992e3545d307c6b9fc5742f065cd9e92f47eec8c` |
| `publication.schema.json` | `af952b628847db9d494b97ec61bef0bfea44189ba8056519ac19a3dfd66381a5` |
| `resource_manifest.schema.json` | `73a43c62f7fffad605432a5cab22c209909c7d3ca5e719569d4b1ff39a314aaa` |
| `selection.schema.json` | `5e1600af74617ce8998e28dd53da90d95ecb7b337ea34952da94be9b79e4bc68` |
| `sources.schema.json` | `2d808a1f81581dd31ef21c7fd3d1f8af28ce00631a710f92b111fa7c9d959e7c` |
| `spine.schema.json` | `e7eee3375ac8dad95fff15f325295e0a84a4786290fd543079bef86733e2f5cd` |
| `take_up.schema.json` | `0e25fe86fe127c5869a84518a74cb96f31097c240db89aa99717515232703353` |
| `vintages.schema.json` | `5bf259074cc6195ce270eeaec3e1f0ada82b920418fecb1cdbab2c6df8b0b4f0` |

“Zero waivers” here can be guaranteed for the operator surface: the only
optional per-run exclusion register in the July command is omitted and loads
as `{}`. Current-main still applies checked-in reviewed exclusions from target
parity (52 entries), release-input coverage (7 entries), eCPS known gaps (7
entries), and hard-coded code authorities. They are source-commit semantics,
not launcher-supplied waivers, and no parser option can replace them with an
empty register. If the owner's zero-waiver ruling includes those code-owned
registers, current main itself is incompatible and that is an additional stop.

### Mutable retry files

A configured checkpoint root may become an input on restart:
`target_frame_checkpoint.h5` is read only when present and identity-equal
(`tools/build_us_fiscal_refresh_release.py:2148-2213,2373-2419`), while
content-addressed target materialization `.json`/`.npy` entries are verified
and read at `tools/build_us_fiscal_refresh_release.py:1748-1801`. They are
generated on cache misses (`tools/build_us_fiscal_refresh_release.py:1804-1834`),
not initial immutable inputs. The requested candidate root does not exist, so
there are no checkpoint bytes or hashes to enumerate and the July checkpoint
was not reused.

## 5. Stage-1 source preflight

Although the stop forbids creating the launcher or running stage 1, all six
pinned raw inputs copied from
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/build-queue.sh:37-53`
were independently found and hashed:

| Input | Measured SHA-256 |
| --- | --- |
| `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5` | `51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe` |
| `/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip` | `8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0` |
| `/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip` | `afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894` |
| `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5` | `0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4` |
| `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5` | `7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df` |
| `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv` | `0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df` |

Their builder requirements are at `tools/build_us_multispine_pool.py:441-514`.
The requested sampling controls parse at
`tools/build_us_multispine_pool.py:531-552`, and output/checkpoint paths derive
from `--out` at `tools/build_us_multispine_pool.py:580-614`.

The off-chain requirement is also code-supported: the pool builder's logbook
predecessor parser and environment fallback are at
`tools/build_us_multispine_pool.py:554-560,4250-4262`. Any eventual invocation
must use `env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST`, omit
`--logbook-prev-row-digest`, and never read or write
`logbook-pending-chain.txt`.

## 6. Scorer contract and absent sparse candidate

The scorer requires `--incumbent`, `--ledger-facts`, and `--out-prefix`; the
candidate path is optional, and a candidate manifest SHA is accepted only with
a pool manifest (`tools/score_us_release_head_to_head.py:260-311,495-515`).
The requested incumbent evidence JSON is:

```text
experiments/replacement_scorecard/incumbent_48b9d479.json
```

Its measured SHA-256 is
`b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8`.
It is evidence, not an H5 accepted by the scorer. The corresponding host
incumbent is:

```text
/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5
```

Its measured SHA-256 is
`48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e`.
No candidate H5 exists. In particular, there is no sparse candidate H5 that
could be named in a second scorer command without fabricating an artifact.

## 7. Runtime and RSS evidence

The best same-week 25% pool receipts are:

- `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/_buildo-runtime/out/stacked-f025-arm-order-r1/build.status.json:44-48`:
  `10150.0256` seconds (2.819 h), `81026367488` bytes (75.46 GiB) maximum
  RSS, nonzero exit.
- `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/_buildo-runtime/out/stacked-f025-arm-investment-r1/build.status.json:44-48`:
  `10075.7451` seconds (2.799 h), `92175892480` bytes (85.85 GiB) maximum
  RSS, nonzero exit.

The roughly 2.8-hour expectation is supported. Roughly 22 GiB is not a safe
peak-RSS expectation; it was at most a transient/top-level observation and is
superseded by these process-tree receipts.

For the July legacy release attempt:

- `/Users/maxghenis/PolicyEngine/_buildo-runtime/logs/buildo-run/chain_densep2.log:16-21`
  records 06:55:23 through 08:34:30, or 1 h 39 m 07 s, ending `rc=1`.
- `/Users/maxghenis/PolicyEngine/_buildo-runtime/logs/buildo-run/pressure_densep2_20260728T065518Z.log:70`
  records a `99149 MiB` (96.83 GiB) peak. The measured pressure-log SHA-256
  is `4579df39aed1f4599a9fb242e057715d5e92a191e0328aba208558ddd1ac46f2`.
- The successful July sparse incumbent receipt at
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/logs/buildo-run/release_sparse.log:538-545`
  records `9723.19` seconds (2 h 42 m 03 s) and `85010432000` bytes
  (79.17 GiB) maximum RSS.

These July figures describe the historical arms and are not a prediction for
the current unified 32,842-target surface. They do show that a 90 GiB
reclaimable-space precondition is unrelated to peak memory safety and that the
requested 22 GiB RSS annotation would be misleading.

## 8. Disposition

Because a mandatory stage-2 input is absent, this round did not create
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/run-candidate.sh`,
run `bash -n`, execute a script `--dry-run`, or synthesize partial stage-1,
stage-2, or scorer command lines. It did not start either builder. It did not
publish, promote, push, tune, touch `logbook-pending-chain.txt`, or write any
logbook predecessor state.

The next owner action is to supply and pin the exact full-SCF bytes accepted
for this legacy release arm, then rule whether the deliverable is one dense
candidate (consistent with `--dense-default-dataset`) or authorizes a second,
separate sparse build. Only after both decisions should script construction
and a real dry-run resume.
