# The release contract

What a Microcosm release must be before anyone may serve it, for every
country. Max ruled ARCH-1 yes on 2026-10-15 (cos d1160), adopting the end state
below. This document states that end state and the five contracts, C1 to C5,
as the code enforces them today and as it will enforce them once publishing
starts refusing non-conforming releases on **2026-11-06**.

Until then nothing here refuses anything. This document, its golden fixtures and
their tests describe and freeze; they do not change what publishes. Each "today"
statement below was read from the code at `origin/main` 84486d083 (2026-10-15) or
executed against it, and cites the function that does the work. Each "from
11-06" statement is a requirement the code does not yet enforce; the milestone
that turns it on is named beside it.

This document replaces `docs/us-uk-release-path.md`, a planning snapshot that
was never merged to main. Its last version (363a9033b, 15 September) lives on the
graph-native branches, which were closed on 2026-10-15 under cos d1161. Two of
its rules carry over: record every reference period explicitly instead of
inferring it from a filename (C1's declared identity below), and make every
dashboard and release verdict refer to the same candidate bytes (C1's digest
bindings).

## The end state

- **One producer per country.** Each country builds one population on
  `run_graph` and derives every published size from it by selection on one
  target register.
- **One release contract.** Every release is certified against C1 to C5, frozen
  at the shape the UK national line already ships.
- **One registry.** `microcosm.data.registry` names every file any consumer
  serves, and policyengine.py's bundle mirrors it.
- **Publishing refuses** any release that does not conform, from 2026-11-06.
- **Retired:** policyengine-uk-data, the Build P patch lanes, the second US
  calibration driver and staging version 1.

The plan that gets there, with owners and dates, is the 10-09 systems report
(`REPORT.md` in the systems-map folder, milestones M0 to M7). This document is
M0's deliverable. Milestones that change what the contract enforces are listed
under [Enforcement timeline](#enforcement-timeline).

## What is served today

Read on 2026-10-15 from policyengine.py's bundle 6.4.3
(`src/policyengine/data/bundle/manifest.json` at d74ddc713), the public
`policyengine/populace-us` Hub repository and `microcosm.data.registry`.

| Grain | Served release | Producer | Meets the contract? |
| --- | --- | --- | --- |
| US national (policyengine.py and `latest.json`) | `populace-us-2024-spm-20260915` | A source enrichment of a parent release (its manifest declares `release_type: source_enrichment` and ships the parent's manifests); the 10-09 report traces the parent to Build P | No: a separate source-enrichment contract, no certification, no Logbook row |
| US state and district (policyengine.py `regional_release_manifest_uri`) | `populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z` | `tools/build_us_acs_local_release.py` | No: not in the registry, no staging, no Logbook row |
| US candidate (tag only) | `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z` (Route A) | `tools/route_a/` then the fiscal-refresh tool | No: 16 gates in an ad hoc `gates` dict, staging `{"enabled": false, "reason": "--no-staging"}`, no certification |
| UK, PolicyEngine app | policyengine-uk-data 1.58.1, `enhanced_frs_2024_25` | policyengine-uk-data | Not a Microcosm release; certified for policyengine-uk 2.130.1 on the `legacy_compatible_model_package` basis |
| UK, `microcosm.data.resolve("uk")` | `populace_uk_2023.h5` (the June 2023 file) | none current | Grandfathered id |
| UK national line | `latest-national.json` names `microcosm-uk-2024-25-national` at a per-cut tag | `microcosm-build-uk` on `run_graph`, then `tools/certify_uk_release_cut.py` and `tools/assemble_uk_release_dir.py` | Meets today's C1 and C5. Lacks what no producer writes yet: C1's declared identity, C3's run-outcome binding and C4's Logbook binding |

So three producers serve three datasets, and microcosm's own registry points the
UK default at a fourth file.

## The five contracts

| | Contract | Freezes | Enforced today by | Golden fixture |
| --- | --- | --- | --- | --- |
| C1 | Release manifest and certification | The release directory's files, both manifests, and a certification composed from gate-battery reports | `validate_release_dir` (`microcosm/data/contract.py`) | `v1/c1/` |
| C2 | Loader compatibility | One rule admitting a model and Core version against a release | `microcosm.data.loader` and policyengine.py, separately | none |
| C3 | Staging telemetry and run outcome | How a build's end is recorded and delivered | `publish_cli._staging_undelivered` | the staging fixtures |
| C4 | Logbook | One durable row per terminal build attempt | nothing at publication | none |
| C5 | Publish and registry | Pointers, the id grammar, and the registry | `prepare_release` / `publish_release` (`microcosm/data/release.py`) and `microcosm.data.registry` | `v1/c5/` |

### C1: the release manifest and certification

#### Today

`validate_release_dir(release_dir)` is the one validator every publish path runs
before uploading anything (`prepare_release` calls it, or its evidence-tier
sibling). It dispatches in three steps:

1. **Release type.** `release_manifest.json`'s `release_type`.
   `source_enrichment` hands off to `validate_source_enrichment_candidate`, a
   separate contract for the Build P inheritance path. The served US default is
   one of these. Any other value except `calibration` is refused.
2. **Dataset role.** `dataset_role` is `national_default` (also when absent) or
   `non_default_local_area`. A local-area release goes to
   `_validate_local_area_release_dir`, or to `_validate_uk_dense_release_dir` when
   its id is the UK dense line's.
3. **The release id string.** Everything else keys on the directory name:
   - `required_release_files(release_id)` adds `us_source_coverage.json` for a
     `populace-us-` prefix, `terminal_gates.json` for a UK exact-k id
     (`populace-uk-<year>-<tier>-k<N>`), and `release_certification.json` for the
     literal `microcosm-uk-2024-25-national`. Any other id gets the three base
     files: `build_manifest.json`, `release_manifest.json` and
     `calibration_diagnostics.json`.
   - `_expected_model_package(release_id)` names `policyengine-us` or
     `policyengine-uk` from the same prefixes. For any other id it returns
     `None`, and the checks then skip the build manifest's model-package
     runtime pin, the built model package's name and the
     `compatible_model_packages` list entirely.
   - The US critical-target fit, stored-input evidence and source-coverage
     checks fire on the `populace-us-` prefix; the UK identity checks fire on
     `populace-uk-` and the exact-k pattern.

Whatever the id, a national release must carry:

- **`build_manifest.json`**: `build_id` equal to the directory name; `code`
  with `repository`, a 40-character `git_commit` and `git_dirty: false`;
  `runtime` with resolved `python`, `policyengine-core` and (when the id names
  one) model-package versions; `dataset` and `calibration` with filenames and
  SHA-256s, the calibration naming its target surface and registry; and a
  `gates` object (`_check_build_manifest`).
- **`release_manifest.json`**: `schema_version` 1
  (`RELEASE_MANIFEST_SCHEMA_VERSION`); `build.build_id` equal to the directory
  name; `build.built_with_core_package` and `build.built_with_model_package`;
  non-empty `compatible_core_packages` and `compatible_model_packages` lists
  that include the built versions; `data_package`; `default_datasets.national`
  naming a `microdata` artifact; and `artifacts`, each with `kind`, `path`,
  `repo_id`, `revision` and `sha256`, all pinned to one revision
  (`_check_release_manifest`).

The **certification** is UK-only. For `microcosm-uk-2024-25-national`,
`release_certification.json` is a required file, and any directory that carries
a national-line gate report without one is refused.
`_check_uk_release_certification` then requires `kind:
uk_release_certification`, `country: uk`, the release id, three signed parts
(`spine`, `calibration_seam`, `release_cut`) whose scoped spec digests match the
pins in `contract.py`, every release-blocking gate passed, and an HMAC-SHA256
attestation over the whole document under the key in
`MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY`.
`_check_uk_certification_evidence_binding` binds each part's report file in the
release directory (`spine_gates.json`, `terminal_gates.json`,
`release_cut_gates.json`) and the score receipt to the signed digests. The
certification is the only place a shippability verdict lives.

#### The id-string keying, and how C1 keys by declared country instead

Every country-specific check hangs off the release id. Two consequences follow,
both executed in `test_release_contract_keying.py`:

- **A new country fails open.** An NZ, BE or AM release, or a future
  `microcosm-us-*` one, gets only the three base files and no model-package
  expectation: no certification, no source coverage, no
  `compatible_model_packages` check. It validates against the weakest contract
  by default.
- **No other country can promote a line.** `line_for_release_id` knows only
  `microcosm-uk-2024-25-national` and `microcosm-uk-2024-25-local-k<N>`. The
  publisher's `--promote-line` and the loader's line reader both call it, so no
  other release can reach a `latest-<line>.json` pointer.

The registry also disagrees with the id about the year. The UK national line is
registered as `("uk", 2025, "national")` while its id says `2024-25`, so the
live-loader assertion in `test_loader.py` that a resolved release id starts with
`microcosm-<country>-<year>-` cannot match the national line once its pointer
resolves (microcosm#1144 reported the failure from a live run).

From 11-06 (M3), C1 dispatches on what the manifest declares instead:

1. **Release manifest schema 2** adds a declared identity, required on every new
   release:

   ```json
   {
     "schema_version": 2,
     "country": "uk",
     "vintage": "2024-25",
     "line": "national",
     "dataset_role": "national_default",
     "producer": {"tool": "tools/assemble_uk_release_dir.py", "commit": "<40-hex>"}
   }
   ```

   `dataset_role` stops being optional. `producer` closes the gap that left Build
   P's producer an inference: a manifest has no tool field today.
2. **One country table in the data shard** replaces the prefix tests. Each
   registered country declares its model package, the files its releases add to
   the C1 base set (the US adds `us_source_coverage.json` and the stored-input
   evidence; every country adds its certification part reports), the lines it may
   publish, and its Hub repository. `required_release_files`,
   `_expected_model_package` and `line_for_release_id` read the table through the
   declared country, never the id.
3. **An unknown country is refused.** The validator fails closed instead of
   falling back to the base set. NZ is registered before the switch (cos d962's
   late-November deliverable depends on it).
4. **The id is derived, not parsed.** A schema-2 release id must equal
   `microcosm-<country>-<vintage>-<line>` rendered from the declared fields, and
   the directory name must equal the id. Nothing reads meaning out of the id
   string any more.
5. **The year is declared once.** The registry entry carries the vintage string,
   and the loader compares the manifest's `vintage` with it exactly, which
   removes the `2025` versus `2024-25` mismatch.
6. **Grandfathering is a frozen list.** Schema-1 manifests keep today's id-keyed
   dispatch only for the release ids published before 11-06. A new schema-1
   release is refused.

#### The frozen shape

C1 freezes what the UK national line ships today. A conforming national or local
release directory carries:

- `build_manifest.json` and `release_manifest.json` in the shape of
  [`v1/c1/microcosm-uk-2024-25-national/`](#golden-fixtures), which
  `uk_national_release_manifests` in `tools/assemble_uk_release_dir.py` writes;
- `calibration_diagnostics.json`;
- `release_certification.json`, composed by the gate battery from the
  country's part reports, with every part's report file beside it bound by
  digest, and the score receipt that decided shippability.

The US meets this by adapter at M2: the fiscal-refresh tool declares its 16
gates in `us/gates.json`, projects its own verdicts through the gate battery
under `BlockingMode.MARKS_ARTIFACT`, and composes the certification from the
projected reports. It projects; it does not re-evaluate. No US release conforms
yet, so there is deliberately no US C1 golden. The US national shape the
publisher accepts until 11-06 stays pinned by `test_support/microcosm_data/release.py`
and the fiscal-refresh builder tests.

### C2: loader compatibility

#### Today: two rules

**microcosm.data's loader.** `_package_certification` reads
`compatible_model_packages` and `compatible_core_packages` from the release
manifest. Each must hold at least one entry for the package with a valid
PEP 440 specifier, and the built version must satisfy one of them. At load,
`_enforce_package_compatibility` refuses unless the installed model package and
the installed `policyengine-core` each satisfy one of their specifiers.

**The release validator.** `_check_compatible_package_entries` checks the same
lists at publication. An entry may declare `basis: publisher_claim` with a
`declared_by` naming who made the claim; an entry without a basis records what
the build measured. It does not require an entry wider than the built version
to declare that basis.

**policyengine.py.** `validate_release_manifest`
(`src/policyengine/provenance/certification.py`) admits a model version when it
equals the built version (basis `built_with_model_package`) or satisfies any
`compatible_model_packages` specifier (basis `legacy_compatible_model_package`).
It tests the model package only. It parses `compatible_core_packages` into its
manifest model but does not test the Core version against it there. Bundle
6.4.3 certifies the UK on the claim basis (built with 2.122.2, certified for
2.130.1) and the US on the built basis (2.2.1).

#### From 11-06: one rule (C2 v1)

For a release and an installed (package, version), where the package is the
declared country's model package or `policyengine-core`:

1. The specifiers are the release manifest's `compatible_<kind>_packages`
   entries for that package. There must be at least one, each valid.
2. The built version satisfies at least one. Otherwise the release is invalid.
3. The installed version satisfies at least one. Otherwise the pair is refused.
4. The basis is `built_with_model_package` when the installed version equals the
   built version. Otherwise it is `publisher_claim`, attributed to the matching
   entry's `declared_by`.
5. An entry that admits any version other than the built one must declare
   `basis: publisher_claim` and `declared_by`. This is the one tightening of the
   validator.

microcosm.data's loader and policyengine.py's bundle certification both
implement this rule. policyengine.py's `legacy_compatible_model_package` label
becomes `publisher_claim`, and it gains the Core check. Owner: anth-volk at M2.
policyengine.py keeps its current rule for policyengine-uk-data releases only
while one is still served.

### C3: staging telemetry and run outcome

#### Today

- **The delivery record.** `build_manifest.json` may carry a `staging` block. In
  version 2 it is exactly `contract_version` (2), `enabled`, `mode`, `run_id`,
  `configured_repository`, `upload_attempts`, `upload_successes`, `read_back`,
  `last_error_code` and `opt_out_reason`. The delivery record stays at version 2
  while the staging documents moved to version 3 (`STAGING_CONTRACT_VERSION` and
  `STAGING_DOCUMENT_VERSION` in `staging_v2.py`).
- **The documents.** Staging version 3 adds a `blocked` lifecycle status beside
  `running`, `completed` and `failed`. The UK producers write it through
  `StagingRunBundleWriterV2`. The US fiscal-refresh tool still writes version 1
  through `microcosm.build.staging.StagingRunBundleWriter`, the only non-test
  importer of version 1. The ACS-local tool writes none.
- **The run outcome.** `microcosm.build.run_outcome` (microcosm#1147) decides how
  a build ended once (`completed`, `blocked`, `failed`, `interrupted`,
  `terminated`, `aborted`) and derives the staging status and the Logbook
  disposition from it. The UK full build uses it; no US tool does.
- **The emitter.** Hosted telemetry is delivered by the external
  `microcosm-emitter` package (pinned `==0.2.1` in microcosm-build), which
  microcosm#1153 extracted on 2026-10-13; Max accepted the extraction on
  2026-10-15 (cos d1162's note). Microcosm keeps a thin adapter,
  `microcosm.build.telemetry_emitter`, that supplies build identity. It is best
  effort: if the service fails to start, the adapter prints a warning and the
  build runs on without telemetry. The fiscal-refresh tool, the UK spine build
  and the UK row-wise staging path start it.
- **The publish gate.** `publish_cli._staging_undelivered` refuses a release
  whose manifest records staging that was intended but never delivered: an empty
  block, a version-2 record in `local_only` mode or with a failed read-back, or
  an enabled run with no successful upload. A disabled record with a non-empty
  reason is a legal opt-out, and so is a manifest with no `staging` key.
  `--allow-missing-staging` overrides the gate. Route A was published with the
  version-1 opt-out.

#### From 11-06 (C3 v1)

- A `national_default` or local-area release may not opt out of staging, and
  `--allow-missing-staging` no longer applies to those roles.
- The build manifest records the build's `run_outcome`, read from its local
  staging version-3 terminal document and bound by that document's digest. The
  publisher refuses any outcome other than `completed`.
- The publisher reads only local records. Emitter delivery is never a
  precondition, so a collector outage cannot block a release.
- Staging version 1, its re-export from `microcosm.build` and the version-1
  branch of `publish_cli` are deleted at M2, once the US writes version 3
  (microcosm#933). The dashboard keeps its version-1 reader for history.

### C4: the Logbook

#### Today

`microcosm.build.logbook.record_build_attempt` validates a row of 17 fields
(`LOGBOOK_ROW_FIELDS`), writes it as `<row_digest>.json` under the caller's
spool directory (file fsync, atomic rename, directory fsync), and only then
attempts a remote insert. Local failures raise; remote failures are returned,
never raised. Rows chain: `row_digest` is the SHA-256 of the canonical JSON of
every non-chain field followed by `prev_row_digest`. The disposition is one of
`BUILD_DISPOSITIONS`, and `run_outcome.logbook_disposition` maps each outcome
onto one.

The UK full build, spine build, row-wise drivers and release-cut certifier write
rows. On the US side only `tools/build_us_multispine_pool.py` and
`tools/us_bundle_generation/core.py` do. Neither the fiscal-refresh tool nor the
ACS-local tool writes one. `logbook/us.jsonl` holds 28 rows, the last from
2026-08-09. No release directory binds a Logbook row today: the UK national
manifests carry the attempt id and cut tag, not a row digest.

#### From 11-06 (C4 v1)

- Every terminal build attempt writes one row through `record_build_attempt`,
  with its disposition from `logbook_disposition(run_outcome)`.
- The release's `build_manifest.json` records its terminal attempt's
  `logbook.row_digest`.
- The publisher refuses a `national_default` or local-area release whose row is
  missing from the local spool or does not hash to that digest. The remote insert
  is not required.
- Whether microcosm#739's Logbook family model folds into `run_outcome` is
  anth-volk's call.

### C5: publish and registry

#### Today

- **Order.** `prepare_release` runs every local guard first, and
  `publish_release` uploads the release files, creates the immutable tag, and
  writes the pointer last, so a reader never sees a pointer before its release.
- **Pointers.** Each is a schema-1 JSON document at the repository root:
  - `latest.json`, the repository-global pointer: `schema_version`,
    `release_id`, `updated_at`, and `paths` to the contract files
    (`latest_pointer_payload`). Only a `national_default` release whose tag is
    its own id may move it.
  - `latest-<line>.json`, a line pointer: the same fields plus `line` and
    `revision`, where the revision may be a per-cut tag
    (`line_pointer_payload`). The line must match `^[a-z]+(-k[1-9][0-9]*)?$`, and
    the release's declared role must be the line's role
    (`dataset_role_for_line`).
  - `latest-evidence.json`, the evidence tier's pointer. The evidence tier is
    being removed (see the retire list).
- **The registry.** `microcosm.data.registry.REGISTRY` maps `(country, year,
  variant)` to a `DatasetSpec` naming the Hub repository, file, engine class and
  pointer path. The loader follows registry, then pointer, then the manifest at
  the pointer's revision, then the artifact, and refuses an artifact whose
  repository or path differs from the registry's. It holds four entries:
  `("uk", 2023, "compact")`, the June file; `("uk", 2025, "national")`, followed
  through `latest-national.json`; `("uk", 2025, "dense")`, tag only; and
  `("us", 2024, "compact")`.
- **The gaps.** The US state and district file that policyengine.py serves has
  no registry entry. The UK default disagrees three ways: the registry's default
  is the June file, policyengine.py serves policyengine-uk-data, and the
  microcosm national line has its own pointer.

#### From 11-06 (C5 v1)

- **One id grammar** for new releases: `microcosm-<country>-<vintage>-<line>`,
  rendered from C1's declared identity. Legacy ids stay on C1's frozen list.
- **Every line publishes through `latest-<line>.json`**, for every country.
  `latest.json` stays as the frozen pointer of the existing US national line
  until that line moves to `latest-national.json`.
- **Every file any consumer serves has a registry entry**, including the US
  state and district file (M2), and each entry declares its vintage.
- **policyengine.py mirrors the registry.** A CI check there asserts that every
  bundle `data_releases` entry resolves through a registry entry (anth-volk, M3).
- **Publishing enforces C1 to C4.** For `national_default` and local-area
  releases the publisher refuses a release without a certification, without a
  local `run_outcome` of `completed`, or without a Logbook row digest, and
  refuses an unknown country.

## The parity rule

The strangler rule from microcosm#378 governs every move of a stage from an
incumbent implementation onto the shared path, including the M4 move of US
national onto the UK national chain and the M6 move of US source stages onto
`run_graph`:

- **The incumbent stays authoritative** until the new stage proves parity on a
  pinned fixture run. Then the stage is deleted from the incumbent.
- **Data, gate and export stages need byte-identical outputs.**
- **Solver stages need declared tolerance bands**, because there is a measured
  nondeterminism floor (the amendment to #378, citing microcosm#307). A solve
  shown deterministic on a platform is held to bitwise-equal weights on that
  platform instead, which is the M4 acceptance for the dense solve.
- **The fixture, the tolerance and the verdict are recorded before the
  deletion**, in the pull request that deletes the incumbent stage.

#378 closes when this document merges; this section supersedes it.

## Enforcement timeline

Owners are the 10-09 report's proposals.

| Milestone | Date | Contract change | Owner |
| --- | --- | --- | --- |
| M0 | 10-16 | This document and the golden fixtures; nothing refuses | Max's session |
| M1 | 10-23 | A UK national cut on policyengine-uk ≥ 2.122.2 with staging v3, a Logbook row and a certification; the registry's UK default flips to it | juaristi22 |
| M2 | 10-30 | `us/gates.json` and the US adapter (C1); the US writes staging v3 and a Logbook row and records `producer` (C3, C4); staging v1 deleted; the US state and district file registered (C5); one compatibility rule in both loaders (C2) | Max's session, PavelMakarchuk, anth-volk |
| M3 | 11-06 | Release manifest schema 2 and declared-country dispatch; unknown countries refused; the publisher refuses non-conforming `national_default` and local-area releases; policyengine.py's registry check | Max's session, juaristi22, anth-volk |

Max's rulings still decide anything that changes published numbers, such as
moving policyengine.py's UK default (UK-1).

## Golden fixtures

`packages/microcosm-data/tests/fixtures/release_contract/v1/` freezes C1 and C5
in bytes, the way `packages/microcosm-build/tests/fixtures/staging/` freezes the
staging documents that calibration-diagnostics vendors and byte-checks.

| File | Contract | Written by |
| --- | --- | --- |
| `c1/microcosm-uk-2024-25-national/build_manifest.json` | C1 | `uk_national_release_manifests`, the UK national assembler's own builder |
| `c1/microcosm-uk-2024-25-national/release_manifest.json` | C1 | the same |
| `c1/inventory.json` | C1 | the contract's constants: required files, the files the assembler ships, and the certification's kind, schema, required fields, part reports and signing-key variable |
| `c5/latest.json` | C5 | `latest_pointer_payload`, serialized as `publish_release` uploads it |
| `c5/latest-national.json` | C5 | `line_pointer_payload`, serialized the same way |
| `c5/registry.json` | C5 | `REGISTRY`, every `DatasetSpec` field |
| `SHA256SUMS` | | the generator |

`tools/generate_release_contract_fixtures.py` writes them, and `--check`
regenerates them into a temporary directory and compares every byte. Every value
is synthetic: each digest is the SHA-256 of `release-contract-fixture:<name>`.

The fixtures carry no gate-battery report and no signed certification, on
purpose. Those embed the UK gate spec's live digests, which change with most UK
pull requests, so a byte-frozen copy would need regenerating almost daily.
`inventory.json` freezes the certification's shape instead, and its signed
validity stays covered by `test_assemble_green_release_dir`, which builds and
validates a full signed national release directory.

The tests:

- `packages/microcosm-data/tests/engine_free/shared/test_release_contract_fixtures.py`:
  checksums pinned; the C1 manifests pass the release contract's manifest checks;
  the loader selects the national artifact from them through the registry's
  national spec; the inventory equals the contract's live constants; both
  pointers equal the publisher's payloads and pass the loader's pointer readers;
  `registry.json` equals the live registry.
- `packages/microcosm-data/tests/engine_free/shared/test_release_contract_keying.py`:
  executable statements of today's id-string keying, which M3 flips
  deliberately.
- `packages/microcosm-build/tests/engine_free/shared/test_release_contract_fixture_generation.py`:
  the generator reproduces the committed bytes.

**Regenerate** when the contract changes on purpose, in the pull request that
changes it and this document:

```bash
uv run python tools/generate_release_contract_fixtures.py
```

A change to the national assembler's manifests, the contract's required files or
certification fields, the pointer payloads or the registry all fail `--check`
until the fixtures are regenerated. That failure is the freeze working. A consumer
that vendors the fixtures (calibration-diagnostics, policyengine.py's registry
check) should pin the SHA-256 of `SHA256SUMS`, as calibration-diagnostics does
for the staging fixtures.

## Retire list, as of 2026-10-15

The 10-09 report's "retire now" items, checked against GitHub and main on
2026-10-15.

| Item | State |
| --- | --- |
| Graph-native drafts #893, #935, #945, #949, #950 | Closed 10-09 |
| #1011, #995, #996, #999, #1002 | Closed 10-15 under cos d1161; branches kept |
| Orrery: #1148 | Closed 10-12; #1144 and #1151 merged |
| #1083 (US process telemetry) | Open (PavelMakarchuk). The emitter now samples the build's process-tree CPU and memory, sends heartbeats and records runtime identity for the fiscal-refresh tool. Its failure classes exist in `run_outcome` but the US tool does not use them yet, and its `work` counters exist nowhere; both belong with microcosm#933 |
| #1082 (US staging always on) | Open (PavelMakarchuk). Belongs in microcosm#933 as staging version 3 for the fiscal-refresh tool |
| #822, #824, #826 (Axiom and Belgium drafts) | Closed by this milestone; BE re-enters through the transport composer after BE-1 |
| #642 (US legislative-district layer) | Closed by this milestone; district products come from the one US population (M5) or #922 |
| #378 (Build K epic) | Closes when this document merges |
| #873 (typed model artifacts) | Kept open: microcosm-dynamics still pins `microcosm-frame` and `microcosm-graph` to its head, 3ff92b0ae, which is not on main. The dynamics work repins or lands it first |
| `tools/build_uk_rowwise_dataset.py` | Kept. The report's premise does not hold: the graph driver accepts `--candidate-clone-counts` only with `--dry-run`, docs/uk-full-build-graph.md keeps the row-wise driver, and UK-3 (dense or local-k) is undecided |
| `format_uk_release_id`, `apply_uk_release_identity` | Deleted by this milestone: no caller outside their own tests and the `uk_runtime` re-export |
| Stale text, microcosm.institute `/calibration`, the evidence tier | Owned by a separate task (task_d2b70813) |
| microcosm-benchmarks, the calibration-diagnostics evaluation harness, policyengine-data's calibrate module | Each lives in another repository; a separate task |
| cos d816 | Ruled 10-09 (per-country Core pins, microcosm#1143) |
| cos d928 | Parked; its 10-10 note records it resolved on main (the plan-2 exclusions renewed to 11-03). Nothing to rule |
