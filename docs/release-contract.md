# The release contract

What a Microcosm release must be before anyone may serve it, for every country.
Max ruled ARCH-1 yes on 2026-10-15 (cos d1160), adopting the end state below,
in which publishing refuses non-conforming releases from 2026-11-06. This
document states that end state and the five contracts, C1 to C5: each as the
code enforces it today, and each one's version 1. For C1 to C4, version 1 is
the requirement the 2026-10-09 plan adds. For C5, version 1 is the build-folder
shape the UK national line ships today, because d1160 froze the contract at the
UK national shape. Main already publishes, in that shape, every release that
`publishes_to_build_folder` names, so C5's version 1 lists what main still
lacks.

Nothing here refuses anything. This document, its golden fixtures and their
tests describe and freeze; they do not change what publishes. Each "today"
statement below comes from reading the code at main's 6c64038f3
(2026-11-12, the merge of microcosm#1583), or from executing against it, and
cites the function that does the work; statements about policyengine.py and the
Hub carry the date of their reading. The golden fixtures, their generator and
their tests arrive with this document, and so does one function named below:
`uk_national_release_manifests` returns the two manifest dictionaries that the
UK national assembler's `_stage_and_finalize` built inline at 6c64038f3. Unless
it says otherwise, each version-1 statement of C1 to C4 is a requirement the
code does not enforce on 2026-11-12. Beside each, this document names the
milestone the plan assigns it to, and the
[enforcement timeline](#enforcement-timeline) records how far each milestone
had got by that date.

The ruling set 2026-11-06 for publishing to start refusing. On 2026-11-12 both
publication commands, `microcosm-publish-release-build` and the tagged
publisher `microcosm-publish-release`, refuse a build that fails today's C1
(`validate_release_dir`, or its evidence sibling) or that recorded staging it
never delivered (today's C3, `publication_refusal`). Max decides when the
version-1 refusals of C1 to C4 start.

This document replaces `docs/us-uk-release-path.md`, a planning snapshot that
never reached main. The snapshot's last version (363a9033b, 2026-09-15) lives
on the graph-native branches. Decision cos d1161 closed the five pull requests
on those branches (#1011, #995, #996, #999 and #1002) on 2026-10-15 and kept
the branches. Two of the snapshot's rules carry over: record every reference
period explicitly instead of inferring it from a filename (C1's declared
identity below), and make every dashboard and release verdict refer to the same
candidate bytes (C1's digest bindings).

## The end state

- **One producer per country.** Each country builds one population on
  `run_graph` and derives every published size from it by selection on one
  target register.
- **One release contract.** Every release passes C1 to C5, frozen at the shape
  the UK national line ships. Each build sits in its own folder,
  `releases/<release>/<build>/`. The repository's `releases.json` records
  every build and names each release's current one. Only
  `microcosm-publish-release-build` publishes a build, and only that command
  makes one current (`--promote`, or `--make-current` at publication). Since
  microcosm#1571 merged on 2026-11-11, the UK national, dense and local-area
  releases and the five US releases publish only this way.
- **One registry.** `microcosm.data.registry` names every file any consumer
  serves, and policyengine.py's bundle mirrors it.
- **Publishing refuses** any release that does not conform (from 2026-11-06, in
  the ruling).
- **Retired:** the incumbent UK data package (uk-data), the Build P patch
  lanes, the second US calibration driver and staging version 1.

The plan that gets there, with owners and dates, is the 2026-10-09 systems
report that cos d1160 cites (Max's planning record, not in this repository;
milestones M0 to M7). This document is M0's deliverable.
[Enforcement timeline](#enforcement-timeline) lists the milestones that change
what the contract enforces.

## Releases in service on 2026-10-15

The table records what three sources said on 2026-10-15: policyengine.py's
bundle 6.4.3 (`src/policyengine/data/bundle/manifest.json` at d74ddc713), the
public `policyengine/populace-us` Hub repository and `microcosm.data.registry`.
The last row names its own source.

| Grain | Served release | Producer | Met the contract on 2026-10-15? |
| --- | --- | --- | --- |
| US national (policyengine.py and `latest.json`) | `populace-us-2024-spm-20260915` | A source enrichment of a parent release (its manifest declares `release_type: source_enrichment` and ships the parent's manifests); the 2026-10-09 report traces the parent to Build P | No: a separate source-enrichment contract, no certification, no Logbook row |
| US state and district (policyengine.py `regional_release_manifest_uri`) | `populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z` | `tools/build_us_acs_local_release.py` | No: not in the registry, no staging, no Logbook row |
| US candidate (tag only) | `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z` (Route A) | `tools/route_a/` then the fiscal-refresh tool | No: 16 gates in an ad hoc `gates` dict, staging `{"enabled": false, "reason": "--no-staging"}`, no certification |
| UK, PolicyEngine app | uk-data 1.58.1, `enhanced_frs_2024_25` | the incumbent UK data package (uk-data) | Not a Microcosm release; certified for policyengine-uk 2.130.1 on the `legacy_compatible_model_package` basis |
| UK, `microcosm.data.resolve("uk")` | `populace_uk_2023.h5` (the June 2023 file) | none current | Grandfathered id |
| UK national line | `latest-national.json` named `microcosm-uk-2024-25-national` at a per-cut tag: the 2026-10-09 report's copy of the pointer, updated 2026-10-04, carries the tag `microcosm-uk-2024-25-national-20261002T230158Z-5c6b3f68`. This document holds no reading of the private UK repository from 2026-10-15 | `microcosm-build-uk` on `run_graph`, then `tools/certify_uk_release_cut.py` and `tools/assemble_uk_release_dir.py` | The assembler refused to write a cut that failed C1 as the code enforced it that day, and line promotion ran that day's C5 guards. It lacked what no producer wrote: C1's declared identity, C3's run-outcome binding and C4's Logbook binding |

So three producers served three datasets, and microcosm's own registry pointed
the UK default at a fourth file.

No row of the table carries a check later than 2026-10-15. Main has since
changed what the registry names and how the loader reaches it:

- On 2026-11-02 microcosm#1432 made the UK 2024-25 national line the UK 2025
  default load. Its registry entry became `("uk", 2025, "compact")`, so
  `microcosm.data.resolve("uk")` resolves to the national line, and the June
  2023 file stays at `("uk", 2023, "compact")`.
- microcosm#1505 (merged 2026-11-05), #1537 (merged 2026-11-09) and #1571
  (merged 2026-11-11) gave every registry entry a release:
  `microcosm-uk-2024-25-national`, `microcosm-uk-2024-25-dense`,
  `populace-uk-2023` (the June 2023 build's release, which
  `publishes_to_build_folder` leaves out, so it takes no new build in a build
  folder) and `microcosm-us-2024-national`. New US national builds declare
  `microcosm_us_2024.h5`, and the entry also accepts `populace_us_2024.h5`, the
  name that the builds published under tags carry.
- For every entry, the loader (`_resolve_certified_release`) reads the
  repository's `releases.json` and loads the release's current build, or a
  listed build that `build=` pins, from `releases/<release>/<build>/`. It
  follows the entry's pointer file (`latest.json` or `latest-national.json`)
  only where the repository has no `releases.json`.
- The registry names no entry for the other four US releases or for any UK
  local-area release, so `microcosm.data.load` reaches none of them.
- The US tools now write builds of standing releases. The fiscal-refresh tool
  (`tools/build_us_fiscal_refresh_release.py`) writes builds of
  `microcosm-us-2024-national` (or of `microcosm-us-2024-evidence`), and
  `tools/build_us_acs_local_release.py` writes builds of
  `microcosm-us-2024-local`.

Main's README says, on 2026-11-12, that the UK and US Hub repositories have
both switched to `releases.json`. This document has not read either
repository.

## The five contracts

| | Contract | Freezes | Enforced today by | Golden fixture |
| --- | --- | --- | --- | --- |
| C1 | Release manifest and certification | The release directory's files, both manifests, and a certification that the certifier composes from gate-battery reports | `validate_release_dir` (`microcosm/data/contract.py`), which `prepare_build_folder_release` runs before every build-folder publication; a build of `microcosm-us-2024-evidence` answers to `validate_evidence_release_dir` instead | `v1/c1/` |
| C2 | Loader compatibility | One rule admitting a model and Core version against a release | `microcosm.data.loader` and policyengine.py, separately | none |
| C3 | Staging telemetry and run outcome | How a build records and delivers its end | `publication_refusal` (`microcosm/data/publish_cli.py`), through `_staging_undelivered`, in both publication commands | the staging fixtures |
| C4 | Logbook | One durable row per terminal build attempt | nothing at publication | none |
| C5 | Publish and registry | The build-folder layout, `releases.json`, release names and the registry | `microcosm-publish-release-build` (`publish_release_build` and `promote_release_build` in `microcosm.build.release_publication`), `prepare_build_folder_release`, `publishes_to_build_folder`, the emitter's release writer, `parse_releases_record` and `microcosm.data.registry` | `v1/c5/` |

### C1: the release manifest and certification

#### Today

`validate_release_dir(release_dir)` holds the certified release contract. Every
command that publishes a build runs it, or its evidence-tier sibling
`validate_evidence_release_dir`, before it hands over a file.
`microcosm-publish-release-build` runs it through
`prepare_build_folder_release` for every build it publishes except a build of
`microcosm-us-2024-evidence`, which answers to the evidence contract instead.
Before it validates anything, the tagged publisher's `prepare_release` refuses
any release that publishes into build folders. Promotion
(`microcosm-publish-release-build --promote`, through `promote_release_build`)
uploads no file and runs no contract: it reads the release and the build that
the local release manifest names.

The validator reads the release's name from the directory. For a release that
publishes each build into its own folder, the directory carries the release's
name, such as `microcosm-us-2024-national`. `publishes_to_build_folder` lists
those releases: the UK national, dense and local-area releases and the five US
releases in `US_RELEASES`. For a build in the tagged layout the directory
carries the build id, such as `populace-us-2024-<sha>-<date>`. Before it
dispatches, the validator refuses two kinds of misplaced build.
`_refuse_misplaced_annual_years` refuses a build-folder build that declares
annual years outside `microcosm-us-2024-annual`, and a build of that release
that declares none. `_refuse_a_receipt_donor_that_is_not_an_enrichment`
refuses a build of `microcosm-us-2024-receipt-donor` that declares any release
type but a source enrichment. The validator then dispatches in three steps:

1. **Release type.** The validator reads `release_manifest.json`'s
   `release_type`. It hands two types to separate contracts and returns.
   `source_enrichment` goes to `validate_source_enrichment_candidate`, the
   Build P inheritance path; the served US default on 2026-10-15 was one of
   these. That contract accepts a build in either layout.
   `compatibility_recertification` goes to `validate_recertification_candidate`
   (`microcosm/data/compatibility_recertification.py`, merged 2026-10-23), the
   contract for a build of `microcosm-us-2024-national` that republishes a
   published US calibration build's bytes under a newer policyengine-us. The
   child always sits in a build folder. It names its parent by release and
   build, or by build id and tag for a parent that kept only its tag. An absent
   `release_type`, or `calibration`, means a calibration build; the validator
   refuses any other value. When the manifest declares an annual-projection
   extension, `validate_annual_projection_extension` checks it next.
2. **Dataset role.** `dataset_role` is `national_default` (also when absent) or
   `non_default_local_area`. A local-area release goes to
   `_validate_local_area_release_dir`, or to `_validate_uk_dense_release_dir`
   when its name is the UK dense release's.
3. **The release name.** Everything else keys on the directory name:
   - `required_release_files(release_id)` adds `us_source_coverage.json` for a
     US release, `terminal_gates.json` for a UK exact-k id
     (`populace-uk-<year>-<tier>-k<N>`), and `release_certification.json` for
     the literal `microcosm-uk-2024-25-national`. `is_us_release` decides the
     first: it accepts the five `US_RELEASES` names and any `populace-us-`
     build id. For any other name `required_release_files` returns the three
     base files: `build_manifest.json`, `release_manifest.json` and
     `calibration_diagnostics.json`. The UK dense and local-area names are
     among them, but step 2 sends a build in the local-area role to its own
     validator and file list first (`_UK_DENSE_REQUIRED_RELEASE_FILES`,
     `LOCAL_AREA_REQUIRED_RELEASE_FILES`). Whatever the name, the validator
     skips `calibration_diagnostics.json` when the release manifest records
     failed diagnostics generation (`calibration_diagnostics.status: failed`,
     which `_calibration_diagnostics_required` reads).
   - `_expected_model_package(release_id)` names `policyengine-us` for a US
     release and `policyengine-uk` for any name that starts `microcosm-uk-` or
     `populace-uk-`. For any other name it returns `None`. The checks then skip
     the build manifest's model-package runtime pin and the
     `compatible_model_packages` list, and they do not check which package
     `built_with_model_package` names (it still needs a non-empty name and a
     resolved version).
   - `publishes_to_build_folder(release_id)` chooses how both manifests name
     the build.
   - The US critical-target fit, stored-input evidence, source-coverage and
     dataset-file checks fire on `is_us_release`; the UK identity checks fire
     on the `populace-uk-` prefix and the exact-k pattern.

Whatever the name, a `calibration` build in the national-default role must
carry:

- **`build_manifest.json`**: `code` with `repository`, a 40-character
  `git_commit` and `git_dirty: false`; `runtime` with resolved `python`,
  `policyengine-core` and, when the name implies a model package, that
  package's versions; `dataset` and `calibration` with filenames and SHA-256s,
  the calibration naming its target surface and registry; and a `gates` object
  (`_check_build_manifest`). In the build-folder layout it also carries
  `release` equal to the directory name and `build`, the build's name, and
  none of the retired fields `build_id`, `attempt_id` and `cut_tag`
  (`_check_build_folder_identity`). The build's name must match the emitter
  interface's identifier rule (`_SAFE_IDENTIFIER_RE`): a letter or digit, then
  up to 127 letters, digits, dots, underscores or hyphens. In the tagged
  layout it carries `build_id` equal to the directory name instead.
- **`release_manifest.json`**: `schema_version` 1
  (`RELEASE_MANIFEST_SCHEMA_VERSION`) in both layouts; a non-empty
  `compatible_core_packages` list that includes the built Core version and,
  when the name implies a model package, a non-empty
  `compatible_model_packages` list that includes the built model version;
  `data_package`; and `default_datasets.national` naming a `microdata`
  artifact (`_check_release_manifest`). In the build-folder layout it also
  carries:
  - `release` and `build` as in the build manifest, and
    `_check_build_folder_builds_agree` refuses two manifests that name
    different builds;
  - `built_at`, `built_with_core_package` and `built_with_model_package` at
    its top level (`_check_build_folder_release_details`);
  - `artifacts`, each with `kind`, a bare file name as `path`, `repo_id`,
    `sha256` and `role`, and no `revision` (`_check_build_folder_artifacts`).
    `release_artifact_role` fixes the role: `dataset` for the microdata,
    `calibration_weights` for the weights, and otherwise the artifact's key.

  A new US build in a build folder declares exactly one microdata file, its
  release's (`US_RELEASE_DATASET_FILES`; `microcosm_us_2024.h5` for a national
  build); the validator holds no tagged `populace-us-` build to that name. An
  annual build declares its base build's H5 and one H5 for each projected year
  instead (`_check_us_release_dataset_file`, `_check_us_annual_dataset_files`).
  In the tagged layout the release manifest carries `build.build_id` equal to
  the directory name, `build.built_with_core_package` and
  `build.built_with_model_package`, and `artifacts` each with `kind`, `path`,
  `repo_id`, `revision` and `sha256`, all pinned to one revision.

A build folder's two manifests therefore differ from the tagged ones in four
ways. They name the release and the build as two strings, where the UK
national line's tagged `build_id` held the release's name and `attempt_id` and
`cut_tag` named the build. The release manifest records the build's details at
its top level. Each artifact carries its role. No artifact pins a revision,
because `releases.json` and the folder locate a build's files. Both layouts
carry `schema_version` 1, and `publishes_to_build_folder` tells them apart. A
build that the switch tool (`microcosm.build.build_folder_switch`) moved from
its tag also carries `migrated_from`, which names the tag, its commit and the
SHA-256 of the manifest the tool rewrote.

Several writers produce the build-folder manifests on main. The UK national and
dense assemblers build both dictionaries themselves and run
`validate_release_dir` before they move the directory into place. The US
fiscal-refresh, ACS-local, source-enrichment and recertification tools build
tagged-shape manifests in a directory named after the release and then call
`rewrite_release_dir` (`microcosm.data.build_folder_manifests`), which rewrites
both manifests in place. The US annual assembler copies its base build's two
manifests, changes `release` and `build` in both, drops `migrated_from`, and in
the release manifest adds the annual metadata and an artifact for each file the
projection adds. The switch tool rewrites the manifests of the builds it moves.

The **certification** is UK-only. The validator requires
`release_certification.json` of `microcosm-uk-2024-25-national`. Whatever its
name, the validator also refuses a calibration build in the national-default
role when its directory carries a `release_cut_gates.json`, or a
`terminal_gates.json` whose `posture` is `calibration_seam`, and no
certification. `_check_uk_release_certification` then requires exactly the
certification's sixteen fields. They include the integer `schema_version` 2,
`kind: uk_release_certification`, `country: uk`, the release id, the parent
spine's digest, and three records: `reference`, `since_baseline` and `parts`.
The `reference` names the build of `microcosm-uk-2024-25-national` that the
certifier compared the cut with: its build, file name, digest and built-with
policyengine-uk version, and whether the cut took the current build or a pinned
one. The `since_baseline` record states what changed since a fixed national
baseline build (`_check_uk_certification_since_baseline`). The `parts` number
exactly three: `spine`, `calibration_seam` and `release_cut`. Each part's
scoped spec digests must match the pins in `contract.py`, and every gate in its
scope must have passed: its `statuses` must equal
`{"passed": <the number of gates in the scope>}`. The document must also say
`shippable: true` and carry, in `spec`, the full gate spec's digests, which
must match the committed full-manifest pins. One HMAC-SHA256 attestation covers
the whole document, under the key in `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY`.
The validator accepts only a key that `microcosm.data.gate_signing_keys`
registers for verification (microcosm#1405, merged 2026-11-09), and the
attestation names that key by its SHA-256. The certification's parts carry no
signatures of their own; each part report beside it carries one, which
`compose_uk_release_certification` verifies before it signs.
`_check_uk_certification_evidence_binding` binds each part's report file in the
release directory (`spine_gates.json`, `terminal_gates.json`,
`release_cut_gates.json`) and the score receipt (`score_vs_reference.json`) to
the signed digests. Schema 2, the reference and the baseline record reached
main with microcosm#1574, merged 2026-11-11. On the national line, a
shippability verdict lives only in the certification.

#### The id-string keying, and how C1 version 1 keys by declared country instead

Every country-specific check hangs off the release name. Two consequences
follow, and `test_release_contract_keying.py` executes the id keying behind
both:

- **A new country fails open.** An NZ, BE or AM release, or a US release of
  another vintage such as `microcosm-us-2025-national`, gets only the three
  base files and no model-package expectation: no certification, no source
  coverage, no `compatible_model_packages` check. It validates against the
  weakest contract by default.
- **No other country can publish into a build folder.**
  `publishes_to_build_folder` names only the UK national, dense and local-area
  releases and the five US releases. `microcosm-publish-release-build` refuses
  any other name, for publication and promotion alike: its `main` checks
  `publishes_to_build_folder` before anything else, and `build_folder_identity`
  refuses the same names inside `publish_release_build` and
  `promote_release_build`. So only the tagged publisher accepts a new country's
  release. `line_for_release_id` still knows only
  `microcosm-uk-2024-25-national` and `microcosm-uk-2024-25-local-k<N>`. Its
  remaining callers serve pointer files: `latest_line_release`, which the
  loader's pointer reader calls, and `line_pointer_payload`, which no publisher
  calls.

The registry also disagrees with the name about the year. The registry files
the UK national release under `("uk", 2025, "compact")`, the UK 2025 default
load since microcosm#1432 (2026-11-02), while its name says `2024-25`; the
entry's `release` field carries that name. `test_release_contract_keying.py`
pins the mismatch. Until microcosm#1432, the live-loader test in
`test_loader.py` required every resolved release id to start with
`microcosm-<country>-<year>-` (or the `populace-` form), which the national
release's name cannot match; microcosm#1144 reported the failure from a live
run. On 2026-11-12 the test requires a release that the loader reads through
`releases.json` to carry the registry entry's release name exactly, and
requires a release that the loader reaches through a line pointer only to start
with `microcosm-<country>-`.

#### Version 1, planned for M3: dispatch on the declared country

Schema 1 covers two layouts on 2026-11-12. microcosm#1505, #1537 and #1571
moved the UK and US releases to the build-folder layout without a schema
bump, so a tagged manifest and a build-folder manifest both carry
`schema_version` 1, and only `publishes_to_build_folder` tells them apart.
Version 1 keeps the schema number as the trigger: a manifest's
`schema_version` says which rules apply to it. Rules that keyed on which
fields a manifest happens to carry would let a half-declared manifest pick its
own rules. The bump to schema 2 lands with the pull request that adds the
declared identity below, and from then the build-folder writers emit 2.

C1 version 1 dispatches on what the manifest declares instead of on its name:

1. **Release manifest schema 2** keeps the build-folder layout and adds a
   declared identity, required on every new build:

   ```json
   {
     "schema_version": 2,
     "release": "microcosm-uk-2024-25-national",
     "build": "<the build's name>",
     "country": "uk",
     "vintage": "2024-25",
     "line": "national",
     "dataset_role": "national_default",
     "producer": {"tool": "tools/assemble_uk_release_dir.py", "commit": "<40-hex>"}
   }
   ```

   `dataset_role` stops being optional. `producer` closes the gap that left
   Build P's producer an inference: no manifest has a field that names its tool
   today. A calibration build's `code` block records the repository and the
   commit, and a source enrichment's or a recertification's also hashes its
   producer's files (`code.source_files_sha256`). The plan has US builds record
   `producer` at M2, and schema 2 requires it of every new build.
2. **One country table in the data shard** replaces the name tests. Each
   registered country declares its model package, the files its releases add to
   the C1 base set (the US adds `us_source_coverage.json` and the stored-input
   evidence; every country adds its certification part reports), its
   country-specific checks (the US critical-target fit, stored-input,
   source-coverage and dataset-file checks; the UK identity, diagnostics and
   gate-report checks), the lines it may publish, and its Hub repository. For
   each line the table gives the role its release takes in `releases.json`
   and the dataset file a build declares, as `US_RELEASES` and
   `US_RELEASE_DATASET_FILES` do for the US today. Every check that keys on a
   name today reads the table through the declared country instead:
   `required_release_files`, `_expected_model_package`, `is_us_release` and
   `publishes_to_build_folder`, and also each exact-k and constant-name test.
   On 2026-11-12 a scan of the syntax tree counts 59 such calls and
   comparisons in `contract.py` and 13 in `release.py`. `loader.py` holds
   none, and `compatibility_recertification.py`, `source_enrichment.py`,
   `annual_projections.py` and `build_folder_manifests.py` hold more.
   `line_for_release_id` leaves with the pointer reader instead. A schema-2 US
   build then cannot skip the US checks by choosing another name.
3. **Unknown countries fail closed.** The validator refuses them instead of
   falling back to the base set. The plan registers NZ before the switch to
   version 1, because cos d962's late-November deliverable depends on it. A new
   country's release joins `publishes_to_build_folder` and the registry in the
   pull request that authorizes its first publication (see C5 version 1).
4. **The validator derives the name instead of parsing it.** A schema-2 release
   name must equal `microcosm-<country>-<vintage>-<line>` rendered from the
   declared fields, and the directory name must equal it. `<line>` comes from
   the country's closed list in the table, which registers each line by exact
   name; C5 version 1 lists the lines in service. Under version 1 nothing reads
   meaning out of the name string. `releases.json` carries no country, vintage
   or line, because the emitter interface fixes its keys. So the loader checks
   the fetched manifest's declared `country`, `vintage` and `line` against the
   registry entry before it loads (planned; no loader does this on 2026-11-12).
5. **One vintage string.** The registry entry carries the vintage string,
   and the loader compares the manifest's `vintage` with it exactly, which
   removes the `2025` versus `2024-25` mismatch.
6. **Grandfathering is a frozen list of published builds.** Schema-1
   manifests keep today's name-keyed dispatch only for the (release, build)
   pairs that each repository's `releases.json` lists at the switch to
   version 1, and for tagged builds that never moved into a folder. The list
   cannot hold release names alone, because a release keeps one constant name
   across builds, so a name-only list would grandfather every future build.
   `microcosm-publish-release-build` refuses a new schema-1 build.

#### The frozen shape

C1 freezes what the UK national release ships today: one build folder,
`releases/microcosm-uk-2024-25-national/<build>/`. A conforming national build
carries:

- `build_manifest.json` and `release_manifest.json` in the shape of
  [`v1/c1/microcosm-uk-2024-25-national/`](#golden-fixtures), which
  `uk_national_release_manifests` in `tools/assemble_uk_release_dir.py` builds;
- `calibration_diagnostics.json`;
- `release_certification.json`, which the release-cut certifier
  (`tools/certify_uk_release_cut.py`, through
  `compose_uk_release_certification`) composes from the gate battery's signed
  part reports;
- every part's report file and the score receipt that compares the cut with its
  reference build, each of which the certification binds by digest.

The UK dense and local-area releases carry no certification today
(`_UK_DENSE_REQUIRED_RELEASE_FILES` and `LOCAL_AREA_REQUIRED_RELEASE_FILES`
name none), and the plan has not said whether they will; C5 version 1 states
the open question for the dense line.

The 2026-10-09 report's plan brings the US to this shape by adapter at M2: the
fiscal-refresh tool declares its gates in `us/gates.json`, projects its own
verdicts through the gate battery under `BlockingMode.MARKS_ARTIFACT`, and
composes the certification from the projected reports. It projects; it does not
re-evaluate. On 2026-11-12 neither `us/gates.json` nor that adapter exists, and
no US build carries a certification. The US national golden,
[`v1/c1/microcosm-us-2024-national/`](#golden-fixtures), therefore freezes
the two manifests the US fiscal-refresh tool writes today for a run that
passes `_build_manifests` only its required arguments and a staging block. A
real build adds blocks that the golden does not freeze. The validator asks
no certification of a US build, so that golden freezes a shape, not a
conforming release.
`test_us_fiscal_refresh_builder.py` and
`test_support/microcosm_data/release.py` (through
`_as_us_national_build_folder_release`) also pin the US national build folder
that `microcosm-publish-release-build` accepts.

### C2: loader compatibility

#### Today: two rules

**microcosm.data's loader.** `_package_certification` reads
`compatible_model_packages` and `compatible_core_packages` from the release
manifest, in either layout. It keeps each entry that names the package and
refuses a specifier that PEP 440 cannot parse. It then refuses the release
unless an entry remains and the built version satisfies one of them. It reads
the built version from a build-folder manifest's top level and from a tagged
manifest's `build` object. At load, `_enforce_package_compatibility` refuses
unless the installed model package and the installed `policyengine-core` each
satisfy one of their specifiers. Only `load` runs that check: `download`,
`resolve_certified` and `download_certified` read and verify the same manifest
and leave the installed versions alone.

**The release validator.** `_check_release_manifest` runs
`_check_compatible_package_entries` on every build with the national role,
whatever its `release_type`: calibration builds, the evidence tier, source
enrichments and recertifications, tagged or in a build folder (where
`_check_build_folder_release_details` runs it). For a release name to which
`_expected_model_package` assigns no package, such as a new country's, the
check skips the model list and still checks the Core list. The local-area
contract runs it on neither list: `_check_local_area_release_manifest` passes
`compatible_packages=False`, so for a local-area or dense build only the loader
checks them. The check requires a non-empty list, a name and a valid specifier
on each entry, an entry for the package, and the built version inside one of
its specifiers. An entry may declare `basis: publisher_claim` with a
`declared_by` naming who made the claim; an entry without a basis records what
the build measured. The check does not require an entry wider than the built
version to declare that basis, and it accepts a Core range.

For source enrichments (the served US default on 2026-10-15),
`validate_source_enrichment_candidate` already asks for more. Each list must
be the exact pin `=={version}` unless the certified report declares a
publisher claim. The validator accepts a claim only as
`compatibility_claim_entry` builds it: it names its declarer, contains the
tested version, and has bounds above and below. Core always takes an exact pin
("no producer can declare a Core range", `source_enrichment.py`).

A `compatibility_recertification` child is a new build of
`microcosm-us-2024-national`. Once its compatibility has passed,
`validate_recertification_candidate` requires both lists to equal what
`claim_entries` builds. `claim_entries` returns every entry of the parent plus
one model claim, which the same `compatibility_claim_entry` builds for the
policyengine-us version the child tested. When no parent entry covers the
tested Core, it adds an exact pin of that Core with `basis: publisher_claim`. A
pending child carries its parent's lists unchanged.
`_check_recertification_claim` narrows the model claim to `==T`, the tested
version, or to `>=L,<U` with `L` above the built version, `L <= T < U`, and `U`
at most the next minor release after `T`. `publish_release_build` refuses to
make a recertification its release's current build, and `promote_release_build`
refuses to promote one, so a loader reaches a child only through `build=`.

**policyengine.py** (its main at 5b6d7901c, read on 2026-11-11).
`validate_release_manifest` (`src/policyengine/provenance/certification.py`)
admits a model version on one of three bases: the version equals the built
version (basis `built_with_model_package`); a `compatible_model_packages`
specifier covers it (basis `legacy_compatible_model_package`); or, given a
`qualified_pair_id`, policyengine.py's own qualification tested that exact
model, Core and data release (basis `qualified_exact_pair`). Given a
`core_version`, `_check_core_compatibility` also requires the Core version to
equal the built Core or to fall under a `compatible_core_packages` claim,
unless the qualified pair covers it. When a certificate follows the bundle's
Core pin instead, `core_claim_warning` reports a pin outside the release's
claims and does not refuse it. Bundle 6.17.2 (the bundle manifest at that
commit) certifies both countries on the built basis: the UK build
`microcosm-uk-2024-25-national-20261107T020434Z-eeb6093e` under policyengine-uk
2.174.1, and the US release `populace-us-2024-spm-20260915` under
policyengine-us 2.2.1. On 2026-10-15, bundle 6.4.3 certified the UK on the
claim basis (built with 2.122.2, certified for 2.130.1) and the US on the built
basis (2.2.1).

#### Version 1, planned for M2: one rule

Version 1 generalizes the source-enrichment rule to every release. For a
release and an installed (package, version), where the package is the declared
country's model package or `policyengine-core`:

1. **Entries.** The rule reads its specifiers from the release manifest's
   `compatible_<kind>_packages` entries for that package. The manifest must
   carry at least one such entry, and each specifier must parse under PEP 440.
2. **Exact pins and claims.** An exact pin is an entry whose specifier is
   exactly `==V` or `===V`, with no wildcard, where
   `Version(V) == Version(built)` (PEP 440 equality, any local label included,
   as in rule 4). Every other entry is a claim. It must declare
   `basis: publisher_claim` and a `declared_by`, must contain the built
   version, and must have bounds above and below. `compatibility_claim_entry`
   applies the same tests today, except that it asks for the tested version,
   which on a source enrichment is the built one.
3. **Core.** `policyengine-core` takes exact pins only. Claims apply to the model
   package alone, as on the source-enrichment path.
4. **Admission.** A loader admits the installed version when
   `Version(installed) == Version(built)`, PEP 440 equality with any local label
   included, or when it satisfies a claim. Containment in an exact pin alone is
   not enough, because `2.0.0+local` satisfies `==2.0.0`. Otherwise the loader
   refuses the pair.
5. **Basis.** A loader names the basis `built_with_model_package` for the first
   case and `publisher_claim` for the second, and attributes the second to the
   first matching claim in manifest order.
6. **Legacy manifests.** Both loaders keep today's rule for a schema-1
   manifest, which every loadable release published before C1 version 1
   carries, and apply rules 1 to 5 to schema-2 manifests. (The evidence tier
   writes `1-evidence`, which microcosm.data's loader refuses.) Adopting C2
   version 1 therefore changes no load of an already published release,
   including the registry's UK default and the releases policyengine.py's
   bundle pins, whatever their entries say.

Every build that `microcosm-publish-release-build` publishes today carries
schema 1, except a build of the evidence release, which carries `1-evidence`
and which microcosm.data's loader refuses. So until C1 version 1 publishes the
first schema-2 manifest, the loaders' half of this rule changes no load, and
rules 2 and 3 tighten the release validator from M2.

Rules 2 and 3 tighten `_check_compatible_package_entries`, which today accepts
a wider entry without a basis and a Core range on every national-role build.
Rule 4 tightens both loaders: microcosm.data's admits any installed version an
entry contains, and policyengine.py's admits any version a claim's specifier
covers.

**Gap: local-area and dense builds.** Version 1 has one publication rule for
every role. A local-area or dense build declares its compatible packages like
any other build, and the validator checks both lists under rules 1 to 3. On
2026-11-12 `_check_local_area_release_manifest` checks neither. C2 version 1
closes the gap at M2, with the rest of the rule (owner: anth-volk).

**Open: the recertification lane.** The `compatibility_recertification` lane
(microcosm#1366, merged 2026-10-23; since microcosm#1571 its children are
builds of `microcosm-us-2024-national`) writes entries that rules 2 and 3, as
written, refuse: a model claim that does not contain the built version, since
the parent's own entries already cover that one, and, when no parent entry
covers the tested Core, an exact pin of that Core with
`basis: publisher_claim`. The release validator meets them first.
`validate_recertification_candidate` runs `_check_release_manifest`, which
reaches `_check_compatible_package_entries` through
`_check_build_folder_release_details`. Rules 2 and 3 tighten that check. A
loader meets them only when a caller loads a child with `build=`, since the
publisher never makes a recertification current. Under rule 6 it then applies
rules 1 to 5 only to the child of a schema-2 parent, because the lane keeps a
child's `schema_version` equal to its parent's. C1 version 1's schema-2
requirement meets the same lane: `_check_release_manifest` compares the child's
schema version with `RELEASE_MANIFEST_SCHEMA_VERSION`, so once that constant
becomes 2, the child of any schema-1 parent fails either that check or the
lane's own equality check. The source-enrichment lane meets the bump too:
`validate_source_enrichment_candidate` requires `schema_version` 1 and runs the
same `_check_release_manifest`, so no source enrichment passes once the
constant becomes 2. Before M2 tightens the validator, either the rules exempt
the recertification lane or that lane changes; the plan does not yet say which.

**Open: policyengine.py's qualified exact pair.** The `qualified_exact_pair`
basis admits a model and Core pair that the manifest neither built with nor
claims, on policyengine.py's own qualification ledger. Rules 4 and 5 have no
such case. Either version 1 names it as a consumer-side basis that no release
can write, or policyengine.py drops it. M2 settles which, when both loaders
adopt the rule.

Under the plan, microcosm.data's loader and policyengine.py's bundle
certification both implement this rule at M2. policyengine.py's
`legacy_compatible_model_package` label becomes `publisher_claim`, and its Core
check becomes unconditional and exact. Owner: anth-volk. On 2026-11-12
microcosm.data's loader still applies today's rule: microcosm#1505 taught
`_package_certification` to read a build folder's manifest and kept its
containment test, and microcosm#1537 and microcosm#1571 left the function
unchanged.

### C3: staging telemetry and run outcome

#### Today

- **The delivery record.** `build_manifest.json` may carry a `staging` block.
  In version 2 the block carries exactly `contract_version` (2), `enabled`,
  `mode`, `run_id`, `configured_repository`, `upload_attempts`,
  `upload_successes`, `read_back`, `last_error_code` and `opt_out_reason`. The
  delivery record stays at version 2 while the staging documents carry version
  3 (`STAGING_CONTRACT_VERSION` and `STAGING_DOCUMENT_VERSION` in
  `staging_v2.py`). The UK national and dense assemblers copy the run's
  `staging_delivery`, which `validate_staging_delivery` checks, into that
  block. When a US tool rewrites its directory for a build folder,
  `build_folder_build_manifest` keeps the block.
- **The documents.** Staging version 3 adds a `blocked` lifecycle status beside
  `running`, `completed` and `failed`. The UK producers write version 3 through
  `StagingRunBundleWriterV2`. The US fiscal-refresh tool still writes version 1
  through `microcosm.build.staging.StagingRunBundleWriter`; no other tool
  imports version 1, and `microcosm.build` re-exports it. The ACS-local tool
  writes no staging documents. Since microcosm#1410 (merged 2026-11-02), both
  writers hand the run's files to the build's emitter, which commits them to
  the staging repository with the build's own Hugging Face credential.
- **The run outcome.** `microcosm.build.run_outcome` (microcosm#1147) decides
  once how a build ended (`completed`, `blocked`, `failed`, `interrupted`,
  `terminated`, `aborted`) and derives the staging status and the Logbook
  disposition from that decision. The UK full build, spine build and row-wise
  staging use it. On the US side only `program_telemetry` calls it, and only to
  label a termination; no US tool takes a staging status or a Logbook
  disposition from it.
- **The emitter.** microcosm#1153 moved the local emitter out of this repository
  into the external `microcosm-emitter` package on 2026-10-13, and Max accepted
  the extraction on 2026-10-15 (cos d1162's note). microcosm-build depends only
  on `microcosm-emitter-interface` (`>=0.14.0,<1` on 2026-11-12). Each machine
  installs the `microcosm-emitter` command on its own (for example with
  `uv tool install microcosm-emitter`). The adapter,
  `microcosm.build.telemetry_emitter`, starts it as the build's child process
  with three modules: telemetry, Orrery graph publication, and the Hugging Face
  hand-over, which commits staging run files and staged dataset bundles. The
  adapter supplies build identity and never starts an emitter older than
  `MINIMUM_EMITTER_VERSION`. `_packaged_minimum_emitter_version` reads that
  floor from the package's `minimum_emitter_version.json` (0.14.0 on
  2026-11-12).
- **Required telemetry.** The emitter runs best effort unless
  `MICROCOSM_TELEMETRY_REQUIRED` says `1`. With the variable unset or at `0`, a
  missing, slow, refusing or outdated emitter leaves the build running without
  it. With `1`, the adapter raises `TelemetryRequiredError` before the build
  does any work when it cannot use the emitter; any other value raises
  `ValueError`. The Modal stage runner (`tools/modal_us_stage_plan.py`) sets
  `1` unless the plan sets `0`, and `tools/modal_us_stage.py` records each
  stage's undelivered telemetry events without refusing anything. No UK driver
  sets the variable, and neither does the fiscal-refresh tool. Delivery to the
  collector gates neither a build nor a release.
- **Who starts an emitter.** The fiscal-refresh tool, the UK spine build, the
  UK row-wise staging path and `program_telemetry` each start a build's own
  emitter; the US Modal programs open `program_telemetry` (the PUF support base
  and the ACS-local release tool among them). Commands that run outside a build
  start a short-lived emitter with only its Hugging Face module
  (`start_bundle_emitter`): `tools/stage_uk_rowwise_candidate.py`,
  `tools/run_uk_cut.py` and `microcosm-publish-release-build`, which hands a
  release build to it.
- **The publish gate.** `publish_cli.publication_refusal` holds the checks
  every publication shares: non-release smoke output, a reform validation that
  skipped its out-of-sample reforms, and undelivered staging
  (`_staging_undelivered`). Both publication commands run it before they check
  the build's release contract: `microcosm-publish-release-build`, once it has
  confirmed that the release publishes into build folders, and
  `microcosm-publish-release`, the tagged publisher. `_staging_undelivered`
  refuses a build whose manifest records staging the build intended but never
  delivered: an empty block, a version-2 record in `local_only` mode or with a
  failed read-back, or an enabled run with no successful upload. It passes a
  disabled record with a non-empty reason as an opt-out. It also passes a
  manifest with no `staging` key, and a US ACS-local build publishes that way.
  `--allow-missing-staging` overrides the gate on both commands. Neither
  `prepare_build_folder_release` nor the function `publish_release_build` runs
  the gate, and promotion (`--promote`) runs none. The Route A candidate in the
  2026-10-15 table carried the version-1 opt-out.

#### Version 1 (planned)

- **Each build answers for what it ran.** C3 and C4 bind a build by what it
  ran. A build that ran a producer ships that run's staging terminal document
  and its Logbook row. A derived build ships a pinned reference to its parent
  (the release, the build and the digests of the parent's two manifests) and
  the outcome record of its own step. It neither copies its parent's staging
  and Logbook records nor inherits them. Three kinds of derived build publish
  today, and each ships this much:
  - **An annual projection** (`tools/assemble_us_annual_release_dir.py`).
    `metadata.annual_projection_base` names the base by repository, release,
    build, file path and the H5's SHA-256, and by no manifest digest. The build
    ships `annual_manifest.json`, `annual_acceptance.json` and one receipt for
    each projected year as its step's records. The assembler copies the base
    build's build manifest, changes `release` and `build` and drops
    `migrated_from`, so the base's `staging` block, when it has one, travels
    into the annual build.
  - **A source enrichment** (`tools/build_us_spm_role_enrichment.py`,
    `tools/build_us_receipt_enrichment_release.py`). `parent_reference` names
    the parent by repository, release and build, with the dataset's SHA-256 and
    the digests that `source_enrichment.py` pins for the parent's files; the
    build carries those files. The SPM-role lane pins both of the parent's
    manifests (`PARENT_FILES`). The receipt lane pins the parent's release
    manifest and not its build manifest (`RECEIPT_PARENT_FILES`), so its
    reference lacks one of the two manifest digests that version 1 asks for. A
    source enrichment's step record is `source_enrichment.json`, and its build
    manifest declares a disabled `staging` block with a reason.
  - **A compatibility recertification**
    (`tools/build_us_compatibility_recertification.py`). The child carries its
    parent's two manifests byte for byte, and
    `compatibility_recertification.json` names the parent by release and
    build and records both manifests' SHA-256. That report is its step's
    record: the parent's contract replayed, and the compatibility verdict. Its
    build manifest declares a disabled `staging` block with a reason.

  No derived build ships a Logbook row today. **Gap for M3:**
  `microcosm-publish-release-build` has no C3 or C4 refusal for a derived
  build. Each lane's own validator checks its parent reference. The staging
  gate has no rule for a derived build: it passes the disabled block that a
  source enrichment or a recertification declares, and it judges an annual
  build by the `staging` block the assembler copied from its base. Version 1
  adds the refusal: a derived build must carry the pinned parent reference and
  its own step's outcome record, and may carry no staging or Logbook record
  that another run wrote.
- A build that ran a producer may not opt out of staging, and
  `--allow-missing-staging` no longer applies to it. Every build that publishes
  into a build folder holds the `national_default` or the local-area role,
  since `RELEASE_ROLES` names no other, so this rule covers every role.
- The publisher keeps requiring remote staging delivery, as it does today,
  because that delivery serves as the pre-publication review: the build appears
  on the staging dashboard before anyone serves it. Telemetry and the Logbook
  differ: their remote halves never gate. A staging outage must not force a
  rebuild, though. M3 adds a command that re-delivers the build's own staging
  bundle and reads it back, before the build publishes. The command writes its
  receipt, `staging_redelivery.json`, into the release directory: the run id,
  the repository, the files committed, the read-back result, and the SHA-256 of
  the `progress.json` it re-delivered. It never rewrites the build manifest.
  When `build_manifest.json`'s `staging` block records an undelivered run,
  `release_manifest.json` names the receipt as an artifact, because a build
  folder publishes only its two manifests, its checksum ledger and the
  artifacts the release manifest declares. The publisher accepts the build only
  if the receipt's read-back passed and its digest matches the shipped
  `progress.json`. M3 also moves the delivery gate from the two commands into
  `prepare_build_folder_release`, so every publication runs it.
- The build folder ships the build's staging version-3 terminal document (its
  `progress.json`) as an artifact that `release_manifest.json` names, and
  `build_manifest.json` binds it by digest. The build manifest records the
  `run_outcome` that document states, and `microcosm-publish-release-build`
  refuses any outcome other than `completed`. Builds run on Modal and other
  hosts while publishing runs elsewhere, so the record travels with the build,
  and the publisher never reads it from the build host.
- The publisher never requires telemetry delivery, so a collector outage cannot
  block a release.
- The plan deletes staging version 1, its re-export from `microcosm.build` and
  the version-1 branch of `_staging_undelivered` at M2, once the US writes
  version 3 (microcosm#933). On 2026-11-12 the fiscal-refresh tool still writes
  version 1, and `microcosm-publish-release-build` gates its builds through
  that branch. Under the plan the staging dashboard keeps a version-1 reader
  for history; this document has not read the dashboard's code.

### C4: the Logbook

#### Today

`microcosm.build.logbook.record_build_attempt` validates a row of 17 fields
(`LOGBOOK_ROW_FIELDS`), writes it as `<row_digest>.json` under the caller's
spool directory (file fsync, atomic rename, directory fsync), and only then
attempts a remote insert. It raises on a local failure; it returns a remote
failure and never raises one. Rows chain: `row_digest` is the SHA-256 of the
canonical JSON of every non-chain field followed by `prev_row_digest`. The
disposition is one of `BUILD_DISPOSITIONS`, and
`run_outcome.logbook_disposition` maps each outcome onto one.

The UK full build, spine build, row-wise drivers and release-cut certifier
write rows, and `tools/run_uk_cut.py` chains the certification's row on the
national build's: it passes the national build's row digest to
`tools/certify_uk_release_cut.py` as `--logbook-prev-row-digest`. On the US
side only `tools/build_us_multispine_pool.py` writes one;
`tools/us_bundle_generation/core.py` reads `logbook/us.jsonl`, validates its
chain and checks one row's `input_pins_digest`. The fiscal-refresh tool, the
ACS-local tool, the annual assembler and `program_telemetry` write none.
`logbook/us.jsonl` holds 28 rows, the last from 2026-08-09.

Only the UK dense release records a Logbook row.
`tools/assemble_uk_dense_release_dir.py` reads the last row file in the
candidate's spool, in file-name order, names the build after the row's
`build_id`, and copies the row's `build_id`, `row_digest` and `prev_row_digest`
into `uk_source_coverage.json`. No validator reads that record, and the row
itself does not ship. The UK national manifests name the build after its
calibration attempt (`build`) and carry no row digest; no US build carries one
either. No publication command reads a Logbook row.

#### Version 1 (planned)

- Every terminal build attempt writes one row through `record_build_attempt`,
  with its disposition from `logbook_disposition(run_outcome)`.
- When `record_build_attempt` returns, the producer copies its terminal
  attempt's row into the release directory as `<row_digest>.json`. It does not
  copy the row later from the spool, because `reconcile_spool` deletes a row
  once the server acknowledges it. `release_manifest.json` names the row as an
  artifact, so it publishes into the build folder, and `build_manifest.json`
  records `logbook.row_digest`. The shipped row replaces the dense release's
  unchecked `logbook_row` record in `uk_source_coverage.json`.
- `microcosm-publish-release-build` refuses a build that ran a producer when
  its shipped row is missing, does not validate as a Logbook row, or does not
  hash to that digest. The publisher does not require the remote insert. A
  derived build ships no row of its own; C3's first rule says what it ships
  instead.
- anth-volk decides whether microcosm#739's Logbook family model folds into
  `run_outcome`.

### C5: publish and registry

#### Today

The UK national, dense and local-area releases and the five US releases
publish each build into its own folder, `releases/<release>/<build>/`. The
repository's `releases.json` records each release's builds and names the
current one. For those releases the `latest*.json` pointer files survive as
read-only legacy: nothing writes one for them, and code reads one only for a
repository that has no `releases.json`.

- **The releases.** `publishes_to_build_folder` names the releases that
  publish into build folders: the UK national release
  `microcosm-uk-2024-25-national`, the dense release
  `microcosm-uk-2024-25-dense`, the local-area releases
  `microcosm-uk-2024-25-local-k<N>`, and the five US releases in
  `US_RELEASES`. The function matches a local-area name by pattern
  (`_UK_LOCAL_LINE_RELEASE_ID_RE`), so it admits every positive `N`.
  `microcosm-us-2024-national` holds every national build:
  calibrated builds, SPM source enrichments, compatibility recertifications
  and exact-k builds. `microcosm-us-2024-local` holds the ACS local-area
  builds, `microcosm-us-2024-receipt-donor` the reported-receipt source
  enrichments, `microcosm-us-2024-evidence` the evidence-tier builds and
  `microcosm-us-2024-annual` the annual projections. `US_RELEASES` gives each
  its role in `releases.json`: `local_area` for the local release and
  `national` for the other four. `RELEASE_ROLES` maps a build's
  `dataset_role` onto those two roles. The registry reads the June 2023 UK
  build as the one build of release `populace-uk-2023`, which
  `publishes_to_build_folder` does not name, so that release takes no new
  build in a folder.
- **Order.** A person runs `microcosm-publish-release-build`
  (`microcosm.build.release_publication`) with the commands the UK assemblers
  and the US tools print. Before it hands anything to an emitter, the command
  refuses, in this order:
  1. a release directory whose name `publishes_to_build_folder` rejects;
  2. what `publication_refusal` refuses: non-release smoke output, a reform
     validation that skipped its out-of-sample reforms, and staging that
     never arrived (each of the last two with its override flag);
  3. `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE`, which the emitter would
     inherit (`_refuse_offline`);
  4. `--make-current` on a compatibility recertification
     (`_refuse_recertification_as_current`);
  5. a release manifest that does not name the directory's release, a build
     and one repository (`build_folder_identity`);
  6. any `latest*.json` pointer file in the repository
     (`_refuse_pointer_files`);
  7. a build that breaks its release contract, names a repository other than
     `--repo-id`, declares a role other than its US release's role in
     `US_RELEASES`, or ships a file whose SHA-256 differs from its manifest
     (`prepare_build_folder_release`);
  8. a recertification whose parent the Hub does not serve as the child
     carries it (`require_parent_publication`), and an annual build whose
     base national build the Hub does not serve
     (`require_base_publication`);
  9. an emitter whose credential cannot write the repository
     (`write_access_problem`).

  `prepare_build_folder_release` ties no UK release to a role: it hands the
  emitter the role that `RELEASE_ROLES` gives the build's own `dataset_role`.
  When the child names its parent by release and build,
  `require_parent_publication` reads the parent from its build folder at
  `main`; it reads a parent that kept only its tag at that tag. The command
  then hands the emitter a `PublishRelease` request: the repository, the
  release, the build, the release's role, whether to make the build current,
  and each file with its role and SHA-256. The emitter's release writer lives
  in the separately installed `microcosm-emitter`; `release_publication`
  documents that it commits the build's folder and the build's `releases.json`
  entry in one commit. This repository holds only the record's reader.
- **What makes a build current.** Publication lists a build without making it
  current. `--promote` makes a published build current
  (`promote_release_build`): it reads only the names in the local manifest,
  refuses a recertification and any pointer file, and hands the emitter a
  `PromoteBuild` request. It runs neither the release contract nor
  `publication_refusal` again. `--make-current` publishes a build and makes
  it current in one change. Both refuse a compatibility recertification,
  which a consumer loads with `build=`. The UK assemblers and
  `publication_commands` print the publish command and the `--promote`
  command.
- **`releases.json`.** `parse_releases_record` reads it under the writer's
  rules: exactly the keys `schema_version` (the integer 1), `default` (a
  listed release or `null`) and `releases`; for each release its `role`, its
  `current` build or `null`, and each build's `published_at` and `withdrawn`;
  a current build that the record lists and has not withdrawn; at most 4 MiB.
  The record carries no country, vintage or line. The one-time `switch` step
  of `microcosm.build.build_folder_switch` writes a repository's first record
  itself and deletes the repository's `latest*.json` files in the same
  commit; after that only the emitter's release writer changes the record.
  The switch's `copy` step moves each tagged build into a folder and rewrites
  its manifests, as C1 describes. Its `delete-old-layout` step then deletes,
  in one commit, the files the tagged layout leaves on `main`: every file
  under `releases/` outside a build folder, and every dataset file (`.h5`,
  `.hdf5`, `.npz`) at the repository's root. It keeps `releases.json`, every
  build folder the record lists, the other root files and every directory
  outside `releases/`, and it refuses a repository without `releases.json`,
  one that still has a `latest*.json` pointer file, and a build folder the
  record does not list. Each tagged build stays readable at its tag, and each
  deleted file stays in `main`'s history.
- **The tagged publisher and the pointer files.** `microcosm-publish-release`
  (`publish_cli`, which `tools/publish_release.sh` wraps) still publishes a
  release that keeps tags. It runs `publication_refusal`, then
  `publish_release`. That function runs `prepare_release` (the release contract
  and the pointer and tag guards), then `require_parent_publication` for a
  recertification, then uploads the release files, creates the immutable tag
  and writes `latest.json` or `latest-evidence.json` last. Each pointer is a
  schema-1 JSON document at the repository root; `latest.json` holds
  `schema_version`, `release_id`, `updated_at`, and `paths` to the contract
  files (`latest_pointer_payload`). Only a `national_default` release whose tag
  is its own id may move `latest.json`, and never a reported-receipt source
  enrichment or a recertification. The tagged publisher refuses by release name
  and by repository state: `prepare_release` refuses every release that
  `publishes_to_build_folder` names, and `publish_release` refuses every
  repository that has `releases.json` (`_refuse_releases_record`; only the
  Hub's answer that the file is absent lets it go on). Nothing writes
  `latest-<line>.json`: `publish_release` takes no line, although
  `line_pointer_payload` remains. Three readers of pointer files remain. The
  loader follows `latest.json` (`latest_release`) or `latest-<line>.json`
  (`latest_line_release`) only for an entry whose repository has no
  `releases.json`. `latest_evidence_release` returns the evidence release's
  current build from `releases.json` where the file exists, and reads
  `latest-evidence.json` otherwise. The switch reads the pointer files once, to
  set each release's current build and the record's `default`.
- **The registry.** `microcosm.data.registry.REGISTRY` maps `(country, year,
  variant)` to a `DatasetSpec`. Each names the Hub repository, the dataset
  file, the engine class, the `release` the loader reads through
  `releases.json`, and the pointer file the loader follows where the
  repository has no `releases.json`. The registry holds four entries:
  - `("uk", 2023, "compact")`: the June 2023 build, release
    `populace-uk-2023`;
  - `("uk", 2025, "compact")`: release `microcosm-uk-2024-25-national`, the
    UK 2025 default load since microcosm#1432 (2026-11-02), with
    `latest-national.json` as its pointer file;
  - `("uk", 2025, "dense")`: release `microcosm-uk-2024-25-dense`;
  - `("us", 2024, "compact")`: release `microcosm-us-2024-national`, whose
    builds declare `microcosm_us_2024.h5` (`build_folder_filename`) or
    `populace_us_2024.h5`.
- **How the loader resolves an entry.** `_resolve_certified_release` reads the
  repository's `releases.json`. Where the file exists,
  `_resolve_build_folder_release` takes the release's current build, or the
  listed build that `build=` pins (a withdrawn pin loads with a warning), and
  reads that build's `release_manifest.json` from its folder.
  `_certified_release_from_manifest` then refuses a manifest that names another
  release or build, declares a role other than the release's role in
  `releases.json`, pins a revision on any artifact, or declares a dataset whose
  role differs from `dataset`, whose repository differs from the registry's, or
  whose file name the entry does not list. The loader goes to the entry's
  pointer file only when the Hub answers that `releases.json` is absent or,
  offline, when the cache never held the record. The loader ignores the
  record's `default`: the registry chooses the release.
  `load(unverified_root=True)` refuses a repository that has `releases.json`.
- **The gaps.** The registry has no entry for the US local-area release
  `microcosm-us-2024-local`, whose builds carry the US state and district
  file, or for any UK `local-k<N>` release. policyengine.py's bundle, read on
  2026-11-11 at its main (5b6d7901c), pins four builds by tag: the UK
  national build `microcosm-uk-2024-25-national-20261107T020434Z-eeb6093e`,
  the UK dense build `microcosm-uk-2024-25-dense-20261107T173210Z-38605444`,
  the US build `populace-us-2024-spm-20260915` and the US local-area build
  `populace-us-2024-buildo-acs-local-766b7c7-20261017T030317Z`. Its certifier
  (`require_tagged_layout`) refuses a tagged manifest from a repository that
  has `releases.json` at `main`. This document does not check what the Hub
  serves. On 2026-10-15 the UK default disagreed three ways: the registry's
  default was the June file, policyengine.py served uk-data, and the
  microcosm national line had its own pointer.

#### Version 1 (planned)

Version 1 is the build-folder shape for every country and line. Max's ruling
cos d1160 froze one contract at the UK national shape, and that shape is the
build folder: each build in `releases/<release>/<build>/`, `releases.json` as
the record, `microcosm-publish-release-build` as the only publisher, and a
change of the current build through that command (`--promote`, or
`--make-current` at publication) as the one step that moves a release. No line
publishes through a `latest-<line>.json` pointer. The 2026-10-09 report's plan
set out to generalize that pointer beyond the UK, when the UK national line
published through `latest-national.json`; the build folder has replaced that
route. Main meets this shape for every release that `publishes_to_build_folder`
names. The rule that every country and line publishes into build folders lists
the gaps that remain in the shape, and the other rules name their own remaining
work.

- **One grammar for release names.** A new release takes the name
  `microcosm-<country>-<vintage>-<line>`, rendered from C1's declared
  identity. `<line>` comes from a closed list for each country, which the
  contract registers by exact name; version 1 writes no pattern for lines.
  The lines in service on 2026-11-12:
  - UK: `national` and `dense`, and a local-area line `local-k<N>` for each
    `N` that the registry names;
  - US: `national`, `local`, `receipt-donor`, `evidence` and `annual`.

  `evidence` is a registered line on the retire list. A UK local-area release
  takes the name `microcosm-uk-2024-25-local-k<N>` for each `N` that the
  registry names, and the pull request that authorizes a k's first
  publication adds its registry entry. On 2026-11-12 the registry names no UK
  local-area release, so the UK list holds no `local-k<N>` line yet. Legacy
  ids, `populace-uk-2023` among them, stay on C1's frozen list. Version 1
  fixes no grammar for build names. A build's name must pass the emitter
  interface's identifier rule (`_SAFE_IDENTIFIER_RE`), and today's forms are
  facts, not rules: a UK national build takes its calibration attempt's id
  (`uk-frs-calibration-attempt-<stamp>-<hex8>`), a UK dense build takes the
  `build_id` of its Logbook row, a US build takes a `populace-us-2024-…` id,
  and an annual build takes `<base build>-annual-<stamp>-<hex8>`.
- **Every country and line publishes into build folders.** A new country's
  release joins `publishes_to_build_folder` and the registry in the pull
  request that authorizes its first publication. The gaps on 2026-11-12:
  - *A new country's tagged publish.* Nothing refuses it: the tagged
    publisher accepts any release outside `publishes_to_build_folder`, into a
    repository without `releases.json`. M3's refusal of unknown countries
    closes the gap, under M3's owners.
  - *The NZ, AM and BE declarations.* Each country's `release_contract.json`
    (under `packages/microcosm-build/src/microcosm/build/`) names
    `latest.json` in its boundary. Each country's owner changes its file
    before that country's first publication.
  - *The UK local-area pattern.* `publishes_to_build_folder` admits
    `microcosm-uk-2024-25-local-k<N>` for every positive `N`, through
    `_UK_LOCAL_LINE_RELEASE_ID_RE`, while the registry names no UK local-area
    release. The function therefore admits a `k` that no pull request
    registered. `test_release_contract_keying.py` pins today's admission with
    `microcosm-uk-2024-25-local-k55000`, which the registry does not name.
    microcosm#1586 plans to narrow the function to the k's that the registry
    names, which flips that case; Max's session owns that change as a code
    pull request. Under that plan `line_for_release_id` keeps the pattern,
    because the pointer reader still calls it.
  - *The tagged publisher.* `microcosm-publish-release` and
    `tools/publish_release.sh` stay installed and sit on the
    [retire list](#retire-list-as-of-2026-10-15). Their removal waits on the
    refusal of a new country's tagged publish. Loading a tagged build that
    never moved into a folder stays.
  - *Pointer readers.* The loader's fallback (`DatasetSpec.pointer_path`,
    `latest_release`, `latest_line_release`) and the `latest-evidence.json`
    fallback in `latest_evidence_release` still resolve through `latest*.json`.
    `microcosm.data` still exports `latest_release`, which reads `latest.json`
    without looking for `releases.json`, and `publish_release`, which writes
    it. They leave once no registered repository lacks `releases.json`. The
    plan names no owner or milestone for them yet.
  - *The UK cut runner.* `tools/run_uk_cut.py` still builds the tagged
    publisher's arguments for the UK national release, and
    `docs/uk-cut-anyone.md` still describes them; `prepare_release` refuses
    that release. Moving the runner to `microcosm-publish-release-build` falls
    to the UK cut's owner.
  - *policyengine.py's pins.* Its bundle, read on 2026-11-11 at 5b6d7901c,
    pins builds by tag; version 1 pins each by release and build in its
    folder.
  - *The default.* The loader takes each country's default release from the
    registry. `releases.json` also names a `default`: in this repository only
    the switch tool sets it, and no loader reads it. The gap: no check at
    publication confirms that the record's `default` agrees with the registry.
  - *The evidence release.* Its builds answer to
    `validate_evidence_release_dir` and fail C1 by design. The
    [retire list](#retire-list-as-of-2026-10-15) names it as an exception:
    version 1 neither admits it to the contract nor retires it.
- **Every file any consumer serves has a registry entry**, including the US
  local-area release `microcosm-us-2024-local` (M2), and each entry declares
  its vintage. On 2026-11-12 the registry has no entry for
  `microcosm-us-2024-local` or for any UK `local-k<N>` release, and
  `DatasetSpec` has no vintage field; each release's name carries the vintage.
- **policyengine.py mirrors the registry.** A CI check there asserts that
  every bundle `data_releases` entry resolves, by release and build, through
  a registry entry (anth-volk, M3).
- **Publishing enforces C1 to C4.** `microcosm-publish-release-build` refuses a
  build without a certification, a build that ran a producer and ships no
  `run_outcome` of `completed` or no Logbook row, a derived build without what
  C3 version 1 asks of it, and an unknown country. On 2026-11-12 it runs the
  release contract (C1) and the staging gate and refuses every release outside
  `publishes_to_build_folder`. It checks no `run_outcome` and no Logbook row,
  and `--promote` runs neither the contract nor the staging gate. The UK dense
  line ships no `release_certification.json`: its signed verdict is its gate
  report, `uk_local_gates.json`. The plan does not yet say whether the dense
  line composes a certification or whether that report counts as one.

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
  bitwise-equal weights on that platform, which is the 2026-10-09 report's M4
  acceptance for the dense solve.

This document adds one requirement to #378's rule: the pull request that
deletes the incumbent stage records the pinned fixture, the declared tolerance
and the parity verdict.

This section supersedes #378's rule. The pull request that adds this document
closes #378.

## Enforcement timeline

The 2026-10-09 report planned these milestones and proposed their owners. The
last column records what main holds at 6c64038f3 on 2026-11-12.

C5's shape reached main outside these milestones. microcosm#1505 (merged
2026-11-05) added `microcosm-publish-release-build`, and microcosm#1534
(merged 2026-11-09) added the tool that switches a repository to build
folders and `releases.json`. microcosm#1537 (merged 2026-11-09) made the
command the only publisher of UK national, dense and local-area builds, and
microcosm#1571 (merged 2026-11-11) did the same for every US build. C5
version 1 takes that shape.

| Milestone | Planned | Contract change | Owner | On main, 2026-11-12 |
| --- | --- | --- | --- | --- |
| M0 | 2026-10-16 | This document and the golden fixtures; nothing refuses | Max's session | microcosm#1285, which adds this document |
| M1 | 2026-10-23 | A UK national cut on policyengine-uk ≥ 2.122.2 with staging version 3, a Logbook row and a certification; the registry's UK default flips to it | juaristi22 | The registry flip, in microcosm#1432 (merged 2026-11-02). Since microcosm#1505 the entry also names release `microcosm-uk-2024-25-national`, and the loader reads that release's current build from `releases.json` where the repository has one. microcosm#1574 (merged 2026-11-11) moved the national certification to schema 2, which adds `reference` and `since_baseline`, and renamed the score receipt to `score_vs_reference.json`. This document does not check the current build's staging, Logbook row or certification on the Hub |
| M2 | 2026-10-30 | `us/gates.json` and the US adapter (C1); the US writes staging version 3 and a Logbook row and records `producer` (C3, C4); staging version 1 goes; the registry gains an entry for the US local-area release, which carries the state and district file (C5); one compatibility rule in both loaders and, at publication, for every role (C2) | Max's session, PavelMakarchuk, anth-volk | None of it: only am, be, nz and uk carry a `gates.json`; the fiscal-refresh tool composes no certification, still writes staging version 1 through `StagingRunBundleWriter` and records no Logbook row; no release manifest records a `producer`; the registry has no entry for `microcosm-us-2024-local`; `_check_local_area_release_manifest` checks no compatibility list; and microcosm.data's loader keeps today's containment rule. policyengine.py, at 5b6d7901c on 2026-11-11, keeps its three bases and the `legacy_compatible_model_package` label |
| M3 | 2026-11-06 | Release manifest schema 2 and declared-country dispatch; the validator refuses unknown countries, which also closes a new country's tagged publish; `microcosm-publish-release-build` refuses non-conforming builds, derived builds included; the staging re-delivery command; policyengine.py's registry check | Max's session, juaristi22, anth-volk | None of it: `RELEASE_MANIFEST_SCHEMA_VERSION` is 1; `required_release_files`, `_expected_model_package` and `publishes_to_build_folder` still key on the release name, and an unknown name still gets the base file set; `microcosm-publish-release-build` runs the nine checks that C5 lists under "Order" and adds none of M3's: no check it runs asks for a `run_outcome`, a Logbook row or a manifest's declared `country`, and `required_release_files` names `release_certification.json` for `microcosm-uk-2024-25-national` alone; C1's other certification rule keys on the gate files a directory carries, not on a declared country. `test_release_contract_keying.py`, which arrives with this document, pins today's keying. This document has not looked for policyengine.py's registry check |

Max's rulings still decide anything that changes published numbers, such as
moving policyengine.py's UK default (UK-1 in the 2026-10-09 report). On
2026-11-11 policyengine.py's bundle, at 5b6d7901c, pins the UK to a build of
the Microcosm national release.

## Golden fixtures

`packages/microcosm-data/tests/fixtures/release_contract/v1/` freezes C1 and C5
in bytes, the way `packages/microcosm-build/tests/fixtures/staging/` freezes the
staging documents that calibration-diagnostics vendors and byte-checks. Each
manifest golden and each `releases.json` holds the bytes that the writer of the
same file in a real build gives for the generator's synthetic inputs, and
`writers.json` names that writer. The folder holds ten files:

| File | Contract | Writers, as `writers.json` names them, in call order (the last one serializes) |
| --- | --- | --- |
| `c1/microcosm-uk-2024-25-national/build_manifest.json` and `release_manifest.json` | C1 | `uk_national_release_manifests`, then `_write_json`, both in `tools/assemble_uk_release_dir.py`. The assembler's `_stage_and_finalize` calls the same two |
| `c1/microcosm-us-2024-national/build_manifest.json` and `release_manifest.json` | C1 | `_build_manifests`, then `_finish_build_folder_release`, both in `tools/build_us_fiscal_refresh_release.py`, then `rewrite_release_dir` (`microcosm.data.build_folder_manifests`), which `_finish_build_folder_release` calls. The builder's `_main` calls the first two |
| `c1/inventory.json` | C1 | The generator's `_c1_inventory`. It records what `microcosm.data.contract` requires of the UK national release: its line, its dataset role and its role in `releases.json`, whether it publishes into build folders, the required files, the retired build fields, the release manifest's schema version, and the certification's file, schema version, kind, required fields, part evidence files, score receipt, gate-battery phases and signing-key variable. It also lists the files the assembler ships, from the assembler's `_EVIDENCE` |
| `c5/uk/releases.json` and `c5/us/releases.json` | C5 | `add_build` and `make_current` (`microcosm_emitter_interface.releases`), `dataclasses.replace`, then `ReleasesRecord.serialize` |
| `c5/registry.json` | C5 | The generator's `_c5_registry`: every field of every `DatasetSpec` in `REGISTRY`, and the registry's default variant |
| `writers.json` | | The generator's `_writers`, which writes the generator's `WRITERS` table |
| `SHA256SUMS` | | The generator. It lists the other nine files |

`tools/generate_release_contract_fixtures.py` writes them. Its `--check`
regenerates them into a temporary directory, compares every byte with the
committed folder and names each file that differs. The generator refuses a run
whose files its `WRITERS` table does not cover exactly.

**The UK national golden.** The generator calls the assembler's pure function,
`uk_national_release_manifests`, with synthetic inputs, and replaces no seam:
the clock, the runtime versions, the commit and each certification part's
status census arrive as arguments. Every digest, package version, commit and
count in the two manifests is synthetic. Each digest is the SHA-256 of
`release-contract-fixture:<name>`, and the build's name,
`uk-frs-calibration-attempt-20260101T000000Z-00000000`, takes the form of a
calibration attempt id. The parts' censuses, `{"passed": 3}`, `{"passed": 2}`
and `{"passed": 4}`, give the build manifest three passing parts without a gate
report. The staging block holds a successful version 2 delivery in the form
that `validate_staging_delivery` returns. The release manifest's SPI Public Use
Tape marker (microcosm#1095) names one synthetic stage. Its study number
(SN 9422) and conditions are the live values that
`microcosm.build.uk_runtime.spi_dependency` writes, which a release that rests
on the tape restates.

**The US national golden.** The generator runs the US builder's own
`_build_manifests` on stub input files, then `_finish_build_folder_release`, so
`rewrite_release_dir` writes the bytes on disk. The builder reads the installed
packages, the checkout and the clock through globals of its module, and
`_build_manifests` computes the calibration gate's verdict itself from the
solve's diagnostics, through `_release_gate_failures`. The generator replaces
four of those globals and nothing else, and its docstring names the four:

- `_runtime_versions` returns synthetic package versions;
- `_git_output` returns one synthetic commit for `rev-parse HEAD`;
- `datetime` becomes a clock fixed at `2026-01-01T00:00:00+00:00`;
- `_release_gate_failures` returns no failures, so the build manifest records
  `gates.calibration.passed: true` and an empty `failures`, as the UK golden
  records passing parts. The synthetic solve then needs no diagnostics.

The stub dataset and calibration files hold `release-contract-fixture:<name>`,
so their digests take the UK golden's form. The three JSON files that the
builder hashes in the release directory (`calibration_diagnostics.json`,
`us_source_coverage.json` and `us_ssi_take_up.json`) each hold
`{"release_contract_fixture": "<file name>"}`, because the builder parses the
coverage file for its fiscal-target exclusion receipt. The golden records that
receipt as absent.

The US manifests embed live constants of the builder on purpose, because the
builder writes them into every build:

- the `gate_evidence` keys (`US_RELEASE_GATE_EVIDENCE_FILES`). Each entry
  reads `not_evaluated`, because the release directory holds no gate's verdict
  and the run skips no gate;
- the documented absent inputs (`US_DOCUMENTED_ABSENT_INPUTS`): five names,
  each with its text;
- the calibration diagnostics' schema version
  (`CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION`, 9);
- the fiscal-target exclusion receipt's key;
- the dataset and calibration keys and file names, and the repository id
  (`policyengine/populace-us`).

The rest of the two US manifests holds what `_build_manifests` writes for the
generator's call, which passes the function only its required arguments and a
staging block. For each optional value the call leaves out, the function
writes its default: no warm start, no selection source, an empty `timing`, a
null `ledger_artifact` and `default_dataset`, and no Medicaid enrollment
substitutions. It also writes no optional gate block, so the build manifest's
`gates` holds only the four entries that the function always writes:
`calibration`, `target_compilation`, `target_profile_coverage` and
`documented_absent_inputs`. A real build's `_main` passes its timing, the
ledger artifact's provenance, its gate results and receipts such as
`base_pool` and `post_export_scoring`, and the golden freezes none of those
blocks.

**The two records.** Each `releases.json` holds one release, the country's
national one, with one build: the build its C1 golden names, published at the
fixture instant, not withdrawn and current. The generator builds each record
with the interface's own functions, in the order a repository lives it.
`add_build(make_current=False)` lists the build, as a publication does.
`make_current` makes it current, as a promotion does.
`dataclasses.replace(default=...)` then names the national release as the
record's `default`, as the switch tool's `_switch_record` does; interface
0.14.0 has no function that sets `default`. `ReleasesRecord.serialize` writes
the bytes. Each release takes the role `national`: the UK's from
`RELEASE_ROLES` and the national line's dataset role, the US's from
`US_RELEASES`.

**What the goldens leave out.** No fixture holds a signed certification or a
gate-battery report. `_check_uk_gate_battery_report` and
`_check_uk_release_certification` refuse either one unless its spec digests
equal the UK gate spec's pins in `contract.py`:
`_UK_GATE_BATTERY_POLICY_SHA256`, `_UK_GATE_BATTERY_GATES_MANIFEST_SHA256`,
`_UK_GATE_BATTERY_SPEC_FINGERPRINT` and, for each certification part,
`_UK_CERTIFICATION_PART_DIGESTS`. Seven merges to main moved the spec
fingerprint between 2026-11-04 and 2026-11-12 (microcosm#1426, #1391, #1511,
#1535, #1550, #1574 and #1583), so a copy frozen before one of them fails those
checks after it. `test_no_fixture_holds_a_signed_certification` checks that no
fixture file is `release_certification.json`, that no document's `kind` is the
certification's, and that no document carries an `attestation` or a
`signature` key at any depth. `inventory.json` freezes the certification's
shape instead. `test_assemble_green_release_dir`, which assembles a national
release directory with a signed certification and validates it, keeps
covering its signed validity.

Each golden build folder holds only its two manifests, and its test pins
exactly the refusals that its omissions draw:

- The UK golden leaves out the dataset, the calibration weights and the six
  evidence files (`calibration_diagnostics.json`, `release_certification.json`,
  `terminal_gates.json`, `release_cut_gates.json`, `spine_gates.json` and
  `score_vs_reference.json`). `validate_release_dir` refuses it for the two
  required files among them, `calibration_diagnostics.json` and
  `release_certification.json`, and for nothing else
  (`test_uk_national_golden_is_refused_only_for_the_files_it_omits`).
- The US golden leaves out the dataset, the calibration weights,
  `calibration_diagnostics.json`, `us_source_coverage.json`,
  `us_ssi_take_up.json` and `stored_inputs.json`. That last file holds the
  stored-input gate's verdict, which carries the SHA-256 of the live
  stored-column register (`register_sha256`). `validate_release_dir` refuses
  the golden for five things
  (`test_us_national_golden_is_refused_only_for_the_files_it_omits`): the two
  missing required files, `calibration_diagnostics.json` and
  `us_source_coverage.json`, then three for the stored-input verdict. Each
  manifest records `stored_inputs` as `not_evaluated`, and the release
  manifest declares no `stored_inputs` artifact.

Both pairs of manifests pass the manifest checks themselves
(`_check_build_manifest`, `_check_release_manifest` and
`_check_cross_manifest_consistency`) with no failure. `inventory.json` and
`registry.json` carry live values on purpose, because they freeze the
contract's constants and the registry.

The tests:

- `packages/microcosm-data/tests/engine_free/shared/test_release_contract_fixtures.py`
  imports `microcosm.data` and neither the emitter interface nor
  `microcosm.build`. It pins every byte through `SHA256SUMS`. It checks that
  `writers.json` names writers for exactly the files that `SHA256SUMS` lists,
  that no golden names the generator among its writers, and that no fixture
  holds a signed certification. It passes both pairs of manifests through the
  manifest checks, pins each golden's refusals and checks the build-folder
  layout of both pairs: release and build as strings, no retired field, and a
  role and no revision on each artifact. It checks that the UK build
  manifest's gates are the three parts' censuses, and that the US build
  manifest records a passing calibration gate with every `gate_evidence` entry
  `not_evaluated`. It compares the inventory with the contract's live
  constants and with the files the UK release manifest ships. It reads each
  `releases.json` with `parse_releases_record` and checks its `default`, its
  one current build and its role against the C1 golden. Through the loader, it
  selects each national artifact from the registry's entry and the record's
  entry, and checks that the loader refuses the build under the other role.
  Last, it compares `registry.json` with the live registry.
- `packages/microcosm-data/tests/engine_free/shared/test_release_contract_keying.py`:
  executable statements of today's name keying, which C1 version 1 replaces
  deliberately.
- `packages/microcosm-build/tests/engine_free/shared/test_release_contract_fixture_generation.py`
  runs the generator's `--check`. It checks that every writer `writers.json`
  names exists, and reads the source to check that the real build path still
  calls each manifest golden's writers: `_stage_and_finalize` calls
  `uk_national_release_manifests` and `_write_json`, the US builder's `_main`
  calls `_build_manifests` and `_finish_build_folder_release`, and
  `_finish_build_folder_release` calls `rewrite_release_dir`. It rebuilds each
  `releases.json` from one publication, one promotion and the `default` step,
  and checks the interface's serialization against the golden byte for byte
  and this repository's reader against the interface's parse. On up to 60
  generated inputs each run, a Hypothesis property checks that for any valid
  input `uk_national_release_manifests` meets the contract: the manifest
  checks pass, the build-folder layout holds, the diagnostics match the
  assembler's six evidence files, the loader selects the build through a
  published and promoted record, the SPI marker and each part's census carry
  through, and the output depends on the inputs alone, key order included.
- `test_assembled_release_has_the_release_contract_golden_shape` in
  `packages/microcosm-build/tests/engine_free/uk/test_uk_release_assembler.py`:
  a real assembly's manifests match the golden key for key and type for type,
  whatever a list's length and whatever provider ids the publisher labels
  carry, and its directory holds exactly the inventory's
  `release_directory_files`.

**Regenerate** when the contract changes on purpose, in the pull request that
changes it and this document:

```bash
uv run python tools/generate_release_contract_fixtures.py
```

Any of these changes fails `--check` until the pull request regenerates the
fixtures:

- the UK national assembler's manifests: what `uk_national_release_manifests`
  writes, `_write_json`'s serialization (indent 1, no final newline), a
  constant they write (the release name, the dataset key and file name, the
  repository id, the evidence files, the SPI marker's study number and
  conditions), or the form `validate_staging_delivery` gives a staging
  delivery;
- what the US builder writes for the generator's call: the blocks that
  `_build_manifests` always writes and the defaults it writes for a value the
  call leaves out, `rewrite_release_dir`'s output (indent 2, a final newline),
  or one of the live constants above. A change to a block that only a real
  build's arguments produce, such as the ledger provenance or an optional
  gate, leaves `--check` green;
- the contract's required files, certification constants, roles, artifact
  roles or retired fields;
- the emitter interface's `add_build`, `make_current` or
  `ReleasesRecord.serialize` (the repository requires
  `microcosm-emitter-interface>=0.14.0,<1`);
- the registry;
- the generator's `WRITERS` table, or one of its synthetic inputs, such as a
  build name, the fixture instant or a runtime version. A change of that kind
  changes the fixtures, not the contract.

That failure is the freeze working. No consumer in this repository vendors
the fixtures. One that does, as policyengine.py's planned registry check
would, should pin the SHA-256 of `SHA256SUMS`, as calibration-diagnostics
pinned the staging fixtures' when this document read it on 2026-11-03.

## Retire list, as of 2026-10-15

The table lists the 2026-10-09 report's "retire now" items as GitHub and main
showed them on 2026-10-15. The last column gives what a second reading found on
2026-11-11, for the rows that reading covered. The rows for the US evidence
release and the tagged publisher joined the table on 2026-11-11.

| Item | State on 2026-10-15 | Read again on 2026-11-11 |
| --- | --- | --- |
| Graph-native drafts #893, #935, #945, #949, #950 | Closed 2026-10-09 | Still closed, unmerged |
| #1011, #995, #996, #999, #1002 | Closed 2026-10-15 under cos d1161; branches kept | Still closed, unmerged; all five branches still exist |
| Orrery: #1148 | Closed 2026-10-12; #1144 and #1151 merged | The same |
| #1083 (US process telemetry) | Open (PavelMakarchuk). The emitter samples the build's process-tree CPU and memory, sends heartbeats and records runtime identity for the fiscal-refresh tool. Its failure classes exist in `run_outcome` but the US tool does not use them yet, and no staging document carries its `work` counters; both belong with microcosm#933. Noted on the pull request for its author | Still open. The fiscal-refresh tool still imports no `run_outcome`, and neither `staging.py` nor `staging_v2.py` carries a `work` key |
| #1082 (US staging always on) | Open (PavelMakarchuk). Belongs in microcosm#933 as staging version 3 for the fiscal-refresh tool. Noted on the pull request for its author | Still open; #933 still open; the tool still writes staging version 1 |
| #822, #824, #826 (Axiom and Belgium drafts) | This milestone closes them; BE re-enters through the transport composer after BE-1 | Closed 2026-10-15, unmerged |
| #642 (US legislative-district layer) | This milestone closes it; district products come from the one US population (M5) or #922 | Closed 2026-10-15, unmerged; #922 open |
| #378 (Build K epic) | Closes when the pull request that adds this document merges | Open |
| #873 (typed model artifacts) | Kept open when checked: microcosm-dynamics then pinned `microcosm-frame` and `microcosm-graph` to 3ff92b0ae, a commit on #873's branch that is not on main. GitHub records #873 closed unmerged at 20:33Z the same day, as superseded; its closing comment says microcosm-dynamics#541 had moved that pin to microcosm main | Still closed, unmerged |
| `tools/build_uk_rowwise_dataset.py` | Kept. The report's reason, that the graph driver already takes `--candidate-clone-counts`, does not settle it: both drivers accept the flag only for a dry run. `docs/uk-full-build-graph.md` keeps the row-wise driver as a separate driver, and Max had not ruled on UK-3 (dense or local-k) by 2026-10-15 | Both drivers still accept the flag only for a dry run |
| `format_uk_release_id`, `apply_uk_release_identity` | This milestone deletes them, with `UKReleaseIdentity`, which builds on `format_uk_release_id`: none had a caller outside its own tests and the `uk_runtime` re-export. The tier vocabulary and the line ids stay | On main none has a caller outside its own tests and the `uk_runtime` re-export |
| Stale text, microcosm.institute `/calibration`, the evidence tier | A separate task owns them (task_d2b70813) | Not read again. For the evidence tier, see the next row |
| The US evidence release, `microcosm-us-2024-evidence` | On this list since 2026-10-15, as the evidence tier | A named exception: version 1 neither admits it to the contract nor retires it. microcosm#1571 made it a standing release in `US_RELEASES`, with the `national` role. `tools/build_us_fiscal_refresh_release.py` writes its builds under `--evidence-release`. `prepare_build_folder_release` checks them with `validate_evidence_release_dir`; they carry schema `1-evidence` and fail C1's certified contract by design. The registry names no entry for it, so `microcosm.data.load` never serves it. No module under `packages/*/src` or `tools/` calls `latest_evidence_release`. Whoever retires the release must first find, from code, the readers of the evidence release and of `latest_evidence_release` |
| The tagged publisher: `microcosm-publish-release` and `tools/publish_release.sh` | Not on the report's list | On the list from 2026-11-11. It refuses every release that `publishes_to_build_folder` names and every repository that has `releases.json`. Its removal waits on the refusal of a new country's tagged publish (C5 version 1, M3). Loading a tagged build that never moved into a folder stays |
| microcosm-benchmarks, the calibration-diagnostics evaluation harness, policyengine-data's calibrate module | Each lives in another repository; proposed as a separate task (task_18c2af98) | Not read again |
| cos d816 | Ruled 2026-10-09 (per-country Core pins, microcosm#1143) | The same |
| cos d928 | Parked; its 2026-10-10 note records it resolved on main (the plan-2 exclusions renewed to 2026-11-03). Nothing to rule | Still parked. On main the plan-2 exclusions read approved on 2026-10-23 and expire on 2027-01-23 |
