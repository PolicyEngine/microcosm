# The release contract

What a Microcosm release must be before anyone may serve it, for every
country. Max ruled ARCH-1 yes on 2026-10-15 (cos d1160), adopting the end state
below, in which publishing refuses non-conforming releases from 2026-11-06.
This document states that end state and the five contracts, C1 to C5: each as
the code enforces it today, and each one's version 1, the requirement the
2026-10-09 plan adds.

Nothing here refuses anything. This document, its golden fixtures and their
tests describe and freeze; they do not change what publishes. Each "today"
statement below comes from reading the code at `origin/main` 4a36edcf3
(2026-11-03), or from executing against it, and cites the function that does the
work; statements about policyengine.py and the Hub carry the date of their
reading. Each version-1 statement is a requirement the code does not enforce on
2026-11-03. Beside each, this document names the milestone the plan assigns it
to, and the [enforcement timeline](#enforcement-timeline) records how far each
milestone had got by that date. When refusal starts is Max's call.

This document replaces `docs/us-uk-release-path.md`, a planning snapshot that
never reached main. Its last version (363a9033b, 15 September) lives on the
graph-native branches, which cos d1161 closed on 2026-10-15. Two of
its rules carry over: record every reference period explicitly instead of
inferring it from a filename (C1's declared identity below), and make every
dashboard and release verdict refer to the same candidate bytes (C1's digest
bindings).

## The end state

- **One producer per country.** Each country builds one population on
  `run_graph` and derives every published size from it by selection on one
  target register.
- **One release contract.** Every release passes C1 to C5, frozen at the shape
  the UK national line already ships.
- **One registry.** `microcosm.data.registry` names every file any consumer
  serves, and policyengine.py's bundle mirrors it.
- **Publishing refuses** any release that does not conform (from 2026-11-06, in
  the ruling).
- **Retired:** the incumbent UK data package (uk-data), the Build P patch lanes, the second US
  calibration driver and staging version 1.

The plan that gets there, with owners and dates, is the 2026-10-09 systems
report that cos d1160 cites (Max's planning record, not in this repository;
milestones M0 to M7). This document is M0's deliverable.
[Enforcement timeline](#enforcement-timeline) lists the milestones that change
what the contract enforces.

## Releases in service on 2026-10-15

Read on 2026-10-15 from policyengine.py's bundle 6.4.3
(`src/policyengine/data/bundle/manifest.json` at d74ddc713), the public
`policyengine/populace-us` Hub repository and `microcosm.data.registry`.

| Grain | Served release | Producer | Meets the contract? |
| --- | --- | --- | --- |
| US national (policyengine.py and `latest.json`) | `populace-us-2024-spm-20260915` | A source enrichment of a parent release (its manifest declares `release_type: source_enrichment` and ships the parent's manifests); the 10-09 report traces the parent to Build P | No: a separate source-enrichment contract, no certification, no Logbook row |
| US state and district (policyengine.py `regional_release_manifest_uri`) | `populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z` | `tools/build_us_acs_local_release.py` | No: not in the registry, no staging, no Logbook row |
| US candidate (tag only) | `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z` (Route A) | `tools/route_a/` then the fiscal-refresh tool | No: 16 gates in an ad hoc `gates` dict, staging `{"enabled": false, "reason": "--no-staging"}`, no certification |
| UK, PolicyEngine app | uk-data 1.58.1, `enhanced_frs_2024_25` | the incumbent UK data package (uk-data) | Not a Microcosm release; certified for policyengine-uk 2.130.1 on the `legacy_compatible_model_package` basis |
| UK, `microcosm.data.resolve("uk")` | `populace_uk_2023.h5` (the June 2023 file) | none current | Grandfathered id |
| UK national line | `latest-national.json` names `microcosm-uk-2024-25-national` at a per-cut tag | `microcosm-build-uk` on `run_graph`, then `tools/certify_uk_release_cut.py` and `tools/assemble_uk_release_dir.py` | The assembler refuses to write a cut that fails today's C1, and line promotion runs C5's guards. It lacks what no producer writes yet: C1's declared identity, C3's run-outcome binding and C4's Logbook binding |

So three producers served three datasets, and microcosm's own registry pointed
the UK default at a fourth file.

This document has not re-read the table since 2026-10-15, except for one change
on main: on 2026-11-02 microcosm#1432 made the UK 2024-25 national line the UK
2025 default load. Its
registry entry is now `("uk", 2025, "compact")`, followed through
`latest-national.json`, so `microcosm.data.resolve("uk")` resolves to the
national line, and the June 2023 file stays at `("uk", 2023, "compact")`.

## The five contracts

| | Contract | Freezes | Enforced today by | Golden fixture |
| --- | --- | --- | --- | --- |
| C1 | Release manifest and certification | The release directory's files, both manifests, and a certification composed from gate-battery reports | `validate_release_dir` (`microcosm/data/contract.py`) | `v1/c1/` |
| C2 | Loader compatibility | One rule admitting a model and Core version against a release | `microcosm.data.loader` and policyengine.py, separately | none |
| C3 | Staging telemetry and run outcome | How a build records and delivers its end | `publish_cli._staging_undelivered` | the staging fixtures |
| C4 | Logbook | One durable row per terminal build attempt | nothing at publication | none |
| C5 | Publish and registry | Pointers, the id grammar, and the registry | `prepare_release` / `publish_release` (`microcosm/data/release.py`) and `microcosm.data.registry` | `v1/c5/` |

### C1: the release manifest and certification

#### Today

`validate_release_dir(release_dir)` is the one validator every publish path runs
before uploading anything (`prepare_release` calls it, or its evidence-tier
sibling). It dispatches in three steps:

1. **Release type.** `release_manifest.json`'s `release_type`. Two types hand
   off to separate contracts and return. `source_enrichment` goes to
   `validate_source_enrichment_candidate`, the Build P inheritance path; the
   served US default on 2026-10-15 was one of these.
   `compatibility_recertification` goes to `validate_recertification_candidate`
   (`microcosm/data/compatibility_recertification.py`, merged 2026-10-23), the
   contract for a release that republishes a published US calibration
   release's bytes under a newer policyengine-us. An absent `release_type`, or
   `calibration`, means a calibration release; the validator refuses any other
   value. When the manifest declares an annual-projection extension,
   `validate_annual_projection_extension` checks it next.
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
     `calibration_diagnostics.json`. Whatever the id, the validator skips
     `calibration_diagnostics.json` when the release manifest records failed
     diagnostics generation (`calibration_diagnostics.status: failed`, read by
     `_calibration_diagnostics_required`), and the pointer payloads then omit its
     path.
   - `_expected_model_package(release_id)` names `policyengine-us` or
     `policyengine-uk` from the same prefixes. For any other id it returns
     `None`. The checks then skip the build manifest's model-package runtime
     pin and the `compatible_model_packages` list entirely, and they no longer
     check which package `build.built_with_model_package` names (it still needs
     a non-empty name and a resolved version).
   - The US critical-target fit, stored-input evidence and source-coverage
     checks fire on the `populace-us-` prefix; the UK identity checks fire on
     `populace-uk-` and the exact-k pattern.

Whatever the id, a `calibration` release in the national-default role must carry:

- **`build_manifest.json`**: `build_id` equal to the directory name; `code`
  with `repository`, a 40-character `git_commit` and `git_dirty: false`;
  `runtime` with resolved `python`, `policyengine-core` and, when the id names
  a model package, that package's versions; `dataset` and `calibration` with filenames and
  SHA-256s, the calibration naming its target surface and registry; and a
  `gates` object (`_check_build_manifest`).
- **`release_manifest.json`**: `schema_version` 1
  (`RELEASE_MANIFEST_SCHEMA_VERSION`); `build.build_id` equal to the directory
  name; `build.built_with_core_package` and `build.built_with_model_package`;
  a non-empty `compatible_core_packages` list that includes the built Core
  version and, when the id names a model package, a non-empty
  `compatible_model_packages` list that includes the built model version;
  `data_package`; `default_datasets.national`
  naming a `microdata` artifact; and `artifacts`, each with `kind`, `path`,
  `repo_id`, `revision` and `sha256`, all pinned to one revision
  (`_check_release_manifest`).

The **certification** is UK-only. For `microcosm-uk-2024-25-national`,
`release_certification.json` is a required file. The validator refuses any
directory that carries a `release_cut_gates.json`, or a `terminal_gates.json`
whose `posture` is `calibration_seam`, without one, whatever its id.
`_check_uk_release_certification` then requires `kind:
uk_release_certification`, `country: uk`, the release id, and exactly three
parts: `spine`, `calibration_seam` and `release_cut`. Each part's scoped spec
digests must match the pins in `contract.py`, and every gate in its scope must
have passed: its `statuses` must equal `{"passed": <gates in the scope>}`. The
document must also say `shippable: true` and carry, in `spec`, the full gate
spec's digests, which must match the committed full-manifest pins. One
HMAC-SHA256 attestation covers the whole document, under the key in
`MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY`; no part carries its own signature.
`_check_uk_certification_evidence_binding` binds each part's report file in the
release directory (`spine_gates.json`, `terminal_gates.json`,
`release_cut_gates.json`) and the score receipt to the signed digests. On the
national line, the certification is the only place a shippability verdict
lives.

#### The id-string keying, and how C1 v1 keys by declared country instead

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

The registry also disagrees with the id about the year. The registry files the
UK national line under `("uk", 2025, "compact")`, the UK 2025 default load since
microcosm#1432 (2026-11-02), while its id says `2024-25`.
`test_release_contract_keying.py` pins the mismatch. Until microcosm#1432, the
live-loader test in `test_loader.py` required every resolved release id to start
with `microcosm-<country>-<year>-` (or the `populace-` form), which the national
line's id cannot match; microcosm#1144 reported the failure from a live run. It
now asks only for `microcosm-<country>-` of a release reached through a line
pointer.

#### Version 1, planned for M3: dispatch on the declared country

C1 v1 dispatches on what the manifest declares instead:

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
   evidence; every country adds its certification part reports), its
   country-specific checks (the US critical-target fit, stored-input,
   source-coverage and split-microdata checks; the UK identity, diagnostics and
   gate-report checks), the lines it may publish, and its Hub repository. Every
   check that keys on an id string today reads the table through the declared
   country instead: `required_release_files`, `_expected_model_package` and
   `line_for_release_id`, and also each `startswith("populace-us-")`,
   exact-k and constant-id test in `contract.py`, `release.py` and `loader.py`
   (about 35 sites). A schema-2 US release then cannot skip the US checks by
   choosing another id.
3. **Unknown countries fail closed.** The validator refuses them instead of
   falling back to the base set. The plan registers NZ before the switch,
   because cos d962's late-November deliverable depends on it.
4. **The validator derives the id instead of parsing it.** A schema-2 release
   id must equal `microcosm-<country>-<vintage>-<line>` rendered from the
   declared fields, and
   the directory name must equal the id. Nothing reads meaning out of the id
   string any more. The pointer readers validate a line before they fetch the
   manifest, so schema-2 pointers also carry `country`, `vintage` and `line`.
   The reader checks them against the registry entry, then against the manifest
   it fetches.
5. **One vintage string.** The registry entry carries the vintage string,
   and the loader compares the manifest's `vintage` with it exactly, which
   removes the `2025` versus `2024-25` mismatch.
6. **Grandfathering is a frozen list of published revisions.** Schema-1
   manifests keep today's id-keyed dispatch only for the (release id, revision)
   pairs published before the switch: the `populace-*` release ids and the
   per-cut tags of the UK national and dense lines. The list cannot hold release
   ids alone, because the UK lines keep one constant id across cuts, so an
   id-only list would grandfather every future cut. The publisher refuses a new
   schema-1 cut.

#### The frozen shape

C1 freezes what the UK national line ships today. A conforming national or local
release directory carries:

- `build_manifest.json` and `release_manifest.json` in the shape of
  [`v1/c1/microcosm-uk-2024-25-national/`](#golden-fixtures), which
  `uk_national_release_manifests` in `tools/assemble_uk_release_dir.py` writes;
- `calibration_diagnostics.json`;
- `release_certification.json`, which the gate battery composes from the
  country's part reports, with every part's report file beside it, bound by
  digest, and the score receipt that decided shippability.

The 10-09 report's plan brings the US to this shape by adapter at M2: the
fiscal-refresh tool declares its 16 gates in `us/gates.json`, projects its own
verdicts through the gate battery under `BlockingMode.MARKS_ARTIFACT`, and
composes the certification from the projected reports. It projects; it does not
re-evaluate. Neither `us/gates.json` nor that adapter exists on 2026-11-03, and
no US release conforms, so there is deliberately no US C1 golden.
`test_support/microcosm_data/release.py` and the fiscal-refresh builder tests
pin the US national shape the publisher accepts today.

### C2: loader compatibility

#### Today: two rules

**microcosm.data's loader.** `_package_certification` reads
`compatible_model_packages` and `compatible_core_packages` from the release
manifest. Each must hold at least one entry for the package with a valid
PEP 440 specifier, and the built version must satisfy one of them. At load,
`_enforce_package_compatibility` refuses unless the installed model package and
the installed `policyengine-core` each satisfy one of their specifiers.

**The release validator.** For `calibration` releases,
`_check_compatible_package_entries` checks the same lists at publication. An
entry may declare `basis: publisher_claim` with a `declared_by` naming who made
the claim; an entry without a basis records what the build measured. It does
not require an entry wider than the built version to declare that basis. For
`source_enrichment` releases (the served US default on 2026-10-15),
`validate_source_enrichment_candidate` is already stricter. Each list must be
the exact pin `=={version}` unless the certified report declares a publisher
claim. The validator accepts a claim only as `compatibility_claim_entry` builds
it: it names its declarer, contains the tested version, and has bounds above and
below. Core always takes an exact pin ("no producer can declare a Core range",
`source_enrichment.py`). For a `compatibility_recertification` child,
`validate_recertification_candidate` requires both lists to equal what
`claim_entries` builds: every entry of the parent, plus one model claim, built
by the same `compatibility_claim_entry`, for the policyengine-us version the
child tested, plus, when no parent entry covers the tested Core, an exact pin of
that Core with `basis: publisher_claim`. `_check_recertification_claim` narrows
the model claim to `==T`, the tested version, or to `>=L,<U` with `L` above the
built version, `L <= T < U`, and `U` at most the next minor release after `T`.

**policyengine.py**, as read on 2026-10-15. `validate_release_manifest`
(`src/policyengine/provenance/certification.py`) admits a model version when it
equals the built version (basis `built_with_model_package`) or satisfies any
`compatible_model_packages` specifier (basis `legacy_compatible_model_package`).
It tests the model package only. It parses `compatible_core_packages` into its
manifest model but does not test the Core version against it there. Bundle
6.4.3 certifies the UK on the claim basis (built with 2.122.2, certified for
2.130.1) and the US on the built basis (2.2.1).

#### Version 1, planned for M2: one rule

It generalizes the source-enrichment rule to every release. For a release and
an installed (package, version), where the package is the declared country's
model package or `policyengine-core`:

1. **Entries.** The specifiers are the release manifest's
   `compatible_<kind>_packages` entries for that package. There must be at least
   one, and each must be a valid PEP 440 specifier.
2. **Exact pins and claims.** An entry whose specifier is exactly `==V` or
   `===V`, with no wildcard, where `Version(V) == Version(built)` (PEP 440
   equality, any local label included, as in rule 4), is an exact pin. Every
   other entry is a claim. It must declare `basis: publisher_claim` and a
   `declared_by`, must contain the built version, and must have bounds above and
   below. `compatibility_claim_entry` applies the same tests today, except that
   it asks for the tested version, which on a source enrichment is the built
   one.
3. **Core.** `policyengine-core` takes exact pins only. Claims apply to the model
   package alone, as on the source-enrichment path.
4. **Admission.** A loader admits the installed version when
   `Version(installed) == Version(built)`, PEP 440 equality with any local label
   included, or when it satisfies a claim. Containment in an exact pin alone is
   not enough, because `2.0.0+local` satisfies `==2.0.0`. Otherwise the loader
   refuses the pair.
5. **Basis.** The basis is `built_with_model_package` for the first case and
   `publisher_claim` for the second, attributed to the first matching claim in
   manifest order.
6. **Legacy manifests.** Both loaders keep today's rule for a schema-1 manifest,
   which every release published before C1 v1 carries, and apply rules 1 to 5
   to schema-2 manifests. Adopting C2 v1 therefore changes no load of an already
   published release, including the registry's UK default and the releases
   policyengine.py's bundle pins, whatever their entries say.

Rules 2 and 3 tighten the validator for `calibration` releases, which today
accepts a wider entry without a basis and a Core range. Rule 4 tightens both
loaders, which today admit any installed version an entry contains.

**Open: the recertification lane.** The `compatibility_recertification` lane
(microcosm#1366, merged 2026-10-23) writes two entries that rules 2 and 3, as
written, refuse: a model claim that does not contain the built version, since
the parent's own entries already cover that one, and an exact pin of the tested
Core with `basis: publisher_claim`. Before a loader adopts C2 v1, either the
rule admits both or the lane changes; the plan does not yet say which.

Under the plan, microcosm.data's loader and policyengine.py's bundle
certification both implement this rule at M2. policyengine.py's
`legacy_compatible_model_package` label becomes `publisher_claim`, and it gains
the Core check. Owner: anth-volk. On 2026-11-03 microcosm.data's loader still
applies today's rule; `loader.py` has not changed since 2026-10-15.
policyengine.py keeps its current rule for uk-data releases only while it still
serves one.

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
  importer of version 1. The ACS-local tool writes none. Since microcosm#1410
  (merged 2026-11-02), both writers hand the run's files to the build's emitter,
  which commits them to the staging repository with the build's own Hugging Face
  credential.
- **The run outcome.** `microcosm.build.run_outcome` (microcosm#1147) decides how
  a build ended once (`completed`, `blocked`, `failed`, `interrupted`,
  `terminated`, `aborted`) and derives the staging status and the Logbook
  disposition from it. The UK full build, spine build and row-wise staging use
  it; no US tool does.
- **The emitter.** microcosm#1153 moved the local emitter out of this repository
  into the external `microcosm-emitter` package on 2026-10-13, and Max accepted
  the extraction on 2026-10-15 (cos d1162's note). Since microcosm#1410,
  microcosm-build depends only on `microcosm-emitter-interface` (`>=0.7.0,<1`).
  Each machine installs the `microcosm-emitter` command on its own (for example
  with `uv tool install microcosm-emitter`), and the adapter,
  `microcosm.build.telemetry_emitter`, starts it as the build's child process
  with three modules: telemetry, Orrery graph publication and the Hugging Face
  hand-over of staging run files. The adapter supplies build identity and never
  starts an emitter older than 0.6.0 (`MINIMUM_EMITTER_VERSION`). It is best
  effort: a missing, slow, refusing or outdated emitter leaves the build running
  without it. The fiscal-refresh tool, the UK spine build and the UK row-wise
  staging path start it.
- **The publish gate.** `publish_cli._staging_undelivered` refuses a release
  whose manifest records staging the build intended but never delivered: an empty
  block, a version-2 record in `local_only` mode or with a failed read-back, or
  an enabled run with no successful upload. A disabled record with a non-empty
  reason is a legal opt-out, and so is a manifest with no `staging` key.
  `--allow-missing-staging` overrides the gate. The Route A candidate in the
  table above carries the version-1 opt-out.

#### Version 1 (planned)

- A `national_default` or local-area release may not opt out of staging, and
  `--allow-missing-staging` no longer applies to those roles.
- Remote staging delivery stays required, as it is today. It is the
  pre-publication review: the build appears on the staging dashboard before
  anyone serves it. That is the deliberate difference from telemetry and the
  Logbook, whose remote halves never gate. A staging outage must not force a
  rebuild, though. M3 adds a command that re-delivers the release's own staging
  bundle and reads it back. The command writes its receipt,
  `staging_redelivery.json`, into the release directory: the run id, the
  repository, the files committed, the read-back result, and the SHA-256 of the
  `progress.json` it re-delivered. It never rewrites the build manifest. When
  `build_manifest.json`'s `staging` block records an undelivered run, the
  receipt joins the release's C1 file set, and the publisher accepts the release
  only if the receipt's read-back passed and its digest matches the shipped
  `progress.json`. M3 also moves the delivery gate from `publish_cli` into
  `prepare_release`, so every publish path runs it.
- The release directory ships the build's staging version-3 terminal document
  (its `progress.json`), which `build_manifest.json` binds by digest and the
  country's C1 file set lists. The build manifest records the `run_outcome` it
  says, and the publisher refuses any outcome other than `completed`. Builds run
  on Modal and other hosts while publishing runs elsewhere, so the record
  travels with the release, and the publisher never reads it from the build
  host.
- Telemetry delivery is never a precondition, so a collector outage cannot
  block a release.
- The plan deletes staging version 1, its re-export from `microcosm.build` and
  the version-1 branch of `publish_cli` at M2, once the US writes version 3
  (microcosm#933). On 2026-11-03 the fiscal-refresh tool still writes version 1.
  The dashboard keeps its version-1 reader for history.

### C4: the Logbook

#### Today

`microcosm.build.logbook.record_build_attempt` validates a row of 17 fields
(`LOGBOOK_ROW_FIELDS`), writes it as `<row_digest>.json` under the caller's
spool directory (file fsync, atomic rename, directory fsync), and only then
attempts a remote insert. Local failures raise; it returns remote failures and
never raises them. Rows chain: `row_digest` is the SHA-256 of the canonical JSON
of every non-chain field followed by `prev_row_digest`. The disposition is one of
`BUILD_DISPOSITIONS`, and `run_outcome.logbook_disposition` maps each outcome
onto one.

The UK full build, spine build, row-wise drivers and release-cut certifier write
rows. On the US side only `tools/build_us_multispine_pool.py` writes one;
`tools/us_bundle_generation/core.py` reads `logbook/us.jsonl` and verifies a
row's digest. Neither the fiscal-refresh tool nor the ACS-local tool writes
one. `logbook/us.jsonl` holds 28 rows, the last from
2026-08-09. No release directory binds a Logbook row today: the UK national
manifests carry the attempt id and cut tag, not a row digest.

#### Version 1 (planned)

- Every terminal build attempt writes one row through `record_build_attempt`,
  with its disposition from `logbook_disposition(run_outcome)`.
- The producer copies its terminal attempt's row into the release directory, as
  `<row_digest>.json`, when `record_build_attempt` returns, not later from the
  spool, because `reconcile_spool` deletes a row once the server acknowledges
  it. `build_manifest.json` records `logbook.row_digest`.
- The publisher refuses a `national_default` or local-area release whose shipped
  row is missing, does not validate as a Logbook row, or does not hash to that
  digest. The publisher does not require the remote insert.
- Whether microcosm#739's Logbook family model folds into `run_outcome` is
  anth-volk's call.

### C5: publish and registry

#### Today

- **Order.** `prepare_release` runs the release contract (`validate_release_dir`)
  and its pointer and tag guards before any upload; `publish_cli` runs
  its reform-validation and staging gates before it calls `prepare_release`.
  `publish_release` then uploads the release files, creates the immutable tag,
  and writes the pointer last, so a reader never sees a pointer before its
  release. For a compatibility recertification it first calls
  `require_parent_publication`, which refuses unless the parent's tag on the
  Hub serves byte for byte the parent manifests the child carries, and every
  parent artifact at its pinned digest.
- **Pointers.** Each is a schema-1 JSON document at the repository root:
  - `latest.json`, the repository-global pointer: `schema_version`,
    `release_id`, `updated_at`, and `paths` to the contract files
    (`latest_pointer_payload`). Only a `national_default` release whose tag is
    its own id may move it, and never a reported-receipt source enrichment or a
    compatibility recertification.
  - `latest-<line>.json`, a line pointer: the same fields plus `line` and
    `revision`, where the revision may be a per-cut tag
    (`line_pointer_payload`). The line must match `^[a-z]+(-k[1-9][0-9]*)?$`, and
    the release's declared role must be the line's role
    (`dataset_role_for_line`).
  - `latest-evidence.json`, the evidence tier's pointer. The 10-09 report lists
    the evidence tier for retirement (see the retire list).
- **The registry.** `microcosm.data.registry.REGISTRY` maps `(country, year,
  variant)` to a `DatasetSpec` naming the Hub repository, file, engine class and
  pointer path. The loader follows registry, then pointer, then the manifest at
  the pointer's revision, then the artifact, and refuses an artifact whose
  repository or path differs from the registry's. It holds four entries:
  `("uk", 2023, "compact")`, the June file; `("uk", 2025, "compact")`, the
  national line, followed through `latest-national.json` and the UK 2025
  default load since microcosm#1432 (2026-11-02); `("uk", 2025, "dense")`, tag
  only; and `("us", 2024, "compact")`.
- **The gaps.** The US state and district file that policyengine.py served on
  2026-10-15 has no registry entry. On 2026-10-15 the UK default disagreed three
  ways: the registry's default was the June file, policyengine.py served
  uk-data, and the microcosm national line had its own pointer. Since
  microcosm#1432 the registry's UK default follows the national line's pointer;
  this document has not re-read policyengine.py's UK default since 2026-10-15.

#### Version 1 (planned)

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
  shipped `run_outcome` of `completed`, or without a shipped Logbook row, and
  refuses an unknown country. The UK dense line ships no
  `release_certification.json`: its signed verdict is its gate report,
  `uk_local_gates.json`. The plan does not yet say whether the dense line
  composes a certification or whether that report counts as one.

## The parity rule

The strangler rule from microcosm#378 governs every move of a stage from an
incumbent implementation onto the shared path, including the M4 move of US
national onto the UK national chain and the M6 move of US source stages onto
`run_graph`:

- **The incumbent stays authoritative** until the new stage proves parity on a
  pinned fixture run. Then the stage leaves the incumbent.
- **Data, gate and export stages need byte-identical outputs.**
- **Solver stages need declared tolerance bands**, because there is a measured
  nondeterminism floor (the amendment to #378, citing microcosm#307). A solve
  that has proved deterministic on a platform must instead produce
  bitwise-equal weights on that platform, which is the 10-09 report's M4
  acceptance for the dense solve.

This document adds one requirement to #378's rule: the pull request that
deletes the incumbent stage records the pinned fixture, the declared tolerance
and the parity verdict.

#378 closes when this document merges; this section supersedes it.

## Enforcement timeline

The 10-09 report planned these milestones; its owners are proposals. The last
column records what main holds on 2026-11-03.

| Milestone | Planned | Contract change | Owner | On main, 2026-11-03 |
| --- | --- | --- | --- | --- |
| M0 | 10-16 | This document and the golden fixtures; nothing refuses | Max's session | This pull request, microcosm#1285 |
| M1 | 10-23 | A UK national cut on policyengine-uk ≥ 2.122.2 with staging v3, a Logbook row and a certification; the registry's UK default flips to it | juaristi22 | The registry flip, in microcosm#1432 (merged 11-02). This document does not check the promoted cut's staging, Logbook row or certification |
| M2 | 10-30 | `us/gates.json` and the US adapter (C1); the US writes staging v3 and a Logbook row and records `producer` (C3, C4); staging v1 goes; the registry gains the US state and district file (C5); one compatibility rule in both loaders (C2) | Max's session, PavelMakarchuk, anth-volk | None of it: no `us/gates.json` or US adapter, the fiscal-refresh tool still writes staging v1 and no Logbook row, the registry has no US state and district entry, and microcosm.data's loader keeps today's rule. This document does not check policyengine.py |
| M3 | 11-06 | Release manifest schema 2 and declared-country dispatch; the validator refuses unknown countries; the publisher refuses non-conforming `national_default` and local-area releases; policyengine.py's registry check | Max's session, juaristi22, anth-volk | None of it: `RELEASE_MANIFEST_SCHEMA_VERSION` is 1, and `test_release_contract_keying.py` passes on today's id keying. This document does not check policyengine.py |

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
regenerates them into a temporary directory and compares every byte. Every
digest, version, identifier and gate count in the C1 manifests and the pointers
is synthetic: each digest is the SHA-256 of `release-contract-fixture:<name>`.
The release manifest's SPI Public Use Tape marker (microcosm#1095) names one
synthetic stage; its study number and conditions are the live constants of
`microcosm.build.uk_runtime.spi_dependency`, which a release that rests on the
tape restates. `inventory.json` and `registry.json` carry live values on
purpose, because they freeze the contract's constants and the registry.

The fixtures carry no gate-battery report and no signed certification, on
purpose. Those embed the UK gate spec's live digests, which change with most UK
pull requests, so a byte-frozen copy would need regenerating almost daily.
`inventory.json` freezes the certification's shape instead, and
`test_assemble_green_release_dir`, which builds and validates a full signed
national release directory, keeps covering its signed validity.

The tests:

- `packages/microcosm-data/tests/engine_free/shared/test_release_contract_fixtures.py`:
  checksums pinned; the C1 manifests pass the release contract's manifest checks;
  the loader selects the national artifact from them through the registry's
  national spec; the inventory equals the contract's live constants; both
  pointers equal the publisher's payloads and pass the loader's pointer readers;
  `registry.json` equals the live registry.
- `packages/microcosm-data/tests/engine_free/shared/test_release_contract_keying.py`:
  executable statements of today's id-string keying, which C1 v1 replaces
  deliberately.
- `packages/microcosm-build/tests/engine_free/shared/test_release_contract_fixture_generation.py`:
  the generator reproduces the committed bytes, and a Hypothesis property holds
  `uk_national_release_manifests` to the contract for every valid input.
- `test_assembled_release_has_the_release_contract_golden_shape` in
  `packages/microcosm-build/tests/engine_free/uk/test_uk_release_assembler.py`:
  a real assembly's manifests match the golden key for key and type for type,
  and its directory holds exactly the inventory's files.

**Regenerate** when the contract changes on purpose, in the pull request that
changes it and this document:

```bash
uv run python tools/generate_release_contract_fixtures.py
```

Any change to the national assembler's manifests, the contract's required files
or certification fields, the pointer payloads or the registry fails `--check`
until the pull request regenerates the fixtures. That failure is the freeze
working. A consumer that vendors the fixtures (calibration-diagnostics,
policyengine.py's registry check) should pin the SHA-256 of `SHA256SUMS`, as
calibration-diagnostics does for the staging fixtures.

## Retire list, as of 2026-10-15

The 10-09 report's "retire now" items, checked against GitHub and main on
2026-10-15.

| Item | State |
| --- | --- |
| Graph-native drafts #893, #935, #945, #949, #950 | Closed 10-09 |
| #1011, #995, #996, #999, #1002 | Closed 10-15 under cos d1161; branches kept |
| Orrery: #1148 | Closed 10-12; #1144 and #1151 merged |
| #1083 (US process telemetry) | Open (PavelMakarchuk). The emitter now samples the build's process-tree CPU and memory, sends heartbeats and records runtime identity for the fiscal-refresh tool. Its failure classes exist in `run_outcome` but the US tool does not use them yet, and its `work` counters exist nowhere; both belong with microcosm#933. Noted on the PR for its author |
| #1082 (US staging always on) | Open (PavelMakarchuk). Belongs in microcosm#933 as staging version 3 for the fiscal-refresh tool. Noted on the PR for its author |
| #822, #824, #826 (Axiom and Belgium drafts) | Closed by this milestone; BE re-enters through the transport composer after BE-1 |
| #642 (US legislative-district layer) | Closed by this milestone; district products come from the one US population (M5) or #922 |
| #378 (Build K epic) | Closes when this document merges |
| #873 (typed model artifacts) | Kept open when checked: microcosm-dynamics then pinned `microcosm-frame` and `microcosm-graph` to 3ff92b0ae, a commit on #873's branch that is not on main. GitHub records #873 closed unmerged at 20:33Z the same day, as superseded; its closing comment says microcosm-dynamics#541 had moved that pin to microcosm main |
| `tools/build_uk_rowwise_dataset.py` | Kept. The report's reason, that the graph driver already takes `--candidate-clone-counts`, does not settle it: both drivers accept the flag only for a dry run. docs/uk-full-build-graph.md keeps the row-wise driver as a separate driver, and UK-3 (dense or local-k) is undecided |
| `format_uk_release_id`, `apply_uk_release_identity` | This milestone deletes them, with `UKReleaseIdentity`, which builds on `format_uk_release_id`: none had a caller outside its own tests and the `uk_runtime` re-export. The tier vocabulary and the line ids stay |
| Stale text, microcosm.institute `/calibration`, the evidence tier | Owned by a separate task (task_d2b70813) |
| microcosm-benchmarks, the calibration-diagnostics evaluation harness, policyengine-data's calibrate module | Each lives in another repository; proposed as a separate task (task_18c2af98) |
| cos d816 | Ruled 10-09 (per-country Core pins, microcosm#1143) |
| cos d928 | Parked; its 10-10 note records it resolved on main (the plan-2 exclusions renewed to 11-03). Nothing to rule |
