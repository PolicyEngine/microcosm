# PROGRESS — microcosm#848 root identity (branch `root-identity-848`)

Lane journal for hash-pinning every raw microdata input and referencing its
Chronicle registration. (The root `PROGRESS.md` belongs to the ACS
predictor-release-join lane; this file follows the `PROGRESS-<lane>.md`
convention already used by `PROGRESS-graph-acceptance.md`.) Historical once
merged — see CLAUDE.md "Root journals are history, not state".

## State

Started 2026-09-02 from origin/main `d2b31496`. Implementation complete; PR
#853 open. 2026-09-02 (later the same day): realigned the Chronicle package
granularity per the cross-PR convention ruling on #853 (see "Convention
realignment" below). Verification green; PR body updated; nothing merged.

## Convention realignment (2026-09-02, cross-PR ruling on #853)

The "Naming chosen" table below documented **one package per distributed
file**. `MaxGhenis` posted a cross-PR ruling on #853 reconciling this with
chronicle#227: Chronicle's `kind: microdata_release` manifests hold a list
under `files[year]`, so **one package per publisher release** is correct —
multiple files share a `package_id`, disambiguated by the existing
`filename` + `sha256` fields already on every `chronicle_artifact` block.
Changed:

| release | old `package_id` | new `package_id` |
| --- | --- | --- |
| FRS 2024-25, 14 tabs | `dwp-frs-2024-25-<tab>` | `dwp-frs-2024-25` |
| WAS round 8 household EUL | `ons-was-round-8-household` | `ons-was-round-8` |
| LCFS 2023-24 household / person | `ons-lcfs-2023-24-{household,person}` | `ons-lcfs-2023-24` |
| ETB 1977-2024 household | `ons-etb-1977-2024-household` | `ons-etb-1977-2024` |

SPI PUT 2022-23 (`hmrc-spi-public-use-tape-2022-23`) and the two US packages
were already one-package-per-release and are unchanged.

**No `ons-lcfs-2018-20` package exists or was created.** The ruling comment's
"LCFS 2023-24 and 2018-20" phrase describes two *vintages appearing under the
`lcfs_consumption` stage's artifacts*, not two LCFS releases: the `2018_20`
vintage tag belongs to the `was_bridge_donor` artifact, which is the same WAS
round-8 file (identical sha256) as the `was_qrf_donor` artifact in the
`was_wealth` stage. Both already shared `package_id` before this change and
both now resolve to `ons-was-round-8` — giving the bridge donor a separate
`ons-lcfs-2018-20` id would have violated invariant 3 (same sha256 must
resolve to the same registration) by registering one file's bytes under two
different packages. Verified directly: only 19 distinct UK sha256 values
carry registrations, matching the ruling's "19 hashes today" count, and no
sha256 resolves to two different `ChronicleArtifactReference` tuples.

Mechanically: `source_manifest.py`'s invariants are keyed on `sha256`, never
on `package_id` uniqueness, so no code, schema, or tooling changed — only
`uk/spec/sources.yaml`, its generated JSON mirror `uk/source_stages.json`
(hand-kept in lockstep; no UK bundle generator exists the way
`generate_us_bundle_from_constants.py` exists for US),
`release_input_coverage_manifest.json` (regenerated via
`tools/build_uk_release_input_coverage_manifest.py`), the one hardcoded
`"dwp/dwp-frs-2024-25-adult"` string in
`test_microdata_root_identity.py:438`, and the UK `spec_sha256` vector in
`test_spec_engine_country_bundles.py` (AM, BE, and the loader golden vector
were unaffected, since no schema/kernel-module code moved — only UK manifest
data did). `DESIGN.md` process rule 5 is convention-agnostic and needed no
edit.

## Contract implemented

Every artifact entry whose `kind` names microdata declares the SHA-256 of the
exact file its stage reads plus a `chronicle_artifact`
`{source_id, package_id, year, sha256, filename, access}`, or carries one
reviewed row in `packages/microcosm-build/src/microcosm/build/microdata_pins_pending.json`.

Three invariants make the reference hard to get wrong:

- the access class is **derived from the kind** (`licensed_microdata` →
  `licensed`, `private_microdata`/`restricted_microdata` → `restricted`,
  `public_microdata` → `public`, `versioned_derived_microdata` → no class, so
  it must be allowlisted);
- the registration year must be one the entry already declares
  (`tax_year_start` exactly, else a four-digit year in `vintage`);
- entries sharing a SHA-256 must resolve to the same registration.

## Naming chosen (for chronicle#221 / chronicle#227) — superseded, see above

Original (2026-09-02 morning) choice, kept here for history — **superseded
same-day** by the "Convention realignment" section above:

Chronicle's live `db/data/*/manifest.yaml` uses a publisher-slug `source_id`
and a kebab-case, publisher-prefixed `package_id`
(`hmrc` / `hmrc-spi-income-bands-2023-24`), and `files[year]` holds exactly one
file per (package, year). So one package per distributed file — ~~this
granularity was wrong; Chronicle's `files[year]` list is precisely what lets
several distributed files share one registration~~:

| release | source_id | package_id (current) | access |
| --- | --- | --- | --- |
| FRS 2024-25 (UKDS SN 9563), 14 tabs | `dwp` | `dwp-frs-2024-25` | licensed |
| WAS round 8 household EUL (SN 7215) | `ons` | `ons-was-round-8` | restricted |
| LCFS 2023-24 household / person | `ons` | `ons-lcfs-2023-24` | restricted |
| ETB 1977-2024 household | `ons` | `ons-etb-1977-2024` | restricted |
| SPI Public Use Tape 2022-23 (SN 9422) | `hmrc` | `hmrc-spi-public-use-tape-2022-23` | restricted |
| CPS ASEC 2023 archive | `census_cps` | `census-cps-asec-2023` | public |
| SCF 2022 summary extract | `federal_reserve` | `federal-reserve-scf-2022-summary-extract` | public |

Note for chronicle#221/#227: the FRS the UK build reads is **2024-25 / SN
9563** (`tax_year_start: 2024`), not 2023-24, and the UK manifest carries
**21** licensed references over 14 distinct tabs, now consolidated to **5**
UK package registrations (plus 2 US) over **19** distinct UK sha256 values.

## Done

- Read CLAUDE.md, DESIGN.md, the Chronicle raw-microdata identity ADR
  (chronicle `origin/adr-raw-microdata-identity`) and chronicle#221's lane log.
- `source_manifest.py`: registration validation, the three invariants,
  `microdata_artifact_entries`, `resolved_chronicle_registrations`,
  `audit_microdata_pins`, and the `MicrodataPinAllowlist` ratchet loader.
- `sources.schema.json`: `chronicle_artifact` definition, with `filename` and
  `access` annotated `operational` to match the artifact-level keys of those
  names.
- Populated registrations: **31 pinned** entries (29 UK + the frozen UK replay
  manifest + 2 US); UK is fully pinned with a zero baseline.
- One shared allowlist with **39** rows (37 US, 1 BE, 1 AM), country-tagged, so
  the ratchet is a single number. A per-country file would have added a
  resource to the AM and BE packages and moved their goldens, which the lane
  brief forbids.
- `source_runtime.py`: fail-closed `verify_microdata_files` (hashes, refuses,
  names publisher/vintage/locator/expected/actual) and
  `verify_recorded_microdata_pins` (cross-checks a producing run's recorded
  pins; fatal on disagreement, reports absence).
- Wired: `build_uk_frs_spine` verifies every tab and licensed input before any
  stage reads a table and records the registrations in its sidecar;
  `load_asec_raw_stage_checkpoint` cross-checks the recorded ASEC pins;
  `build_us_fiscal_refresh_release` writes `microdata_registrations` next to
  the Chronicle consumer-artifact pin.
- `test_microdata_root_identity.py`: 36 contract tests (`shared-spec` group,
  `tools/ci_test_groups.py --verify` green).
- Re-pinned the byte pins that legitimately moved with the manifests: the US
  `stage_asset` digest and its two test copies, the UK frozen replay digest,
  the regenerated `uk/release_input_coverage_manifest.json`, and four
  `spec_sha256` vectors that attest the schema set.
- DESIGN.md process rule 5; `changelog.d/848-root-identity.added.md`.

## Gotcha for the next editor of this lane

`microcosm.build.source_runtime` is a `_DIRECT_KERNEL_MODULE` in
`spec_engine/seeds.py`, so its **source** is attested by every country's
`spec_sha256`. Any edit to the gate's code moves the `am`, `be`, `uk`, and
loader golden vectors, even when no manifest changed. Re-pin those four last,
after the code is final, or you will chase them repeatedly.

## Next

- Push the convention-realignment commits, edit the #853 PR body to state the
  package-per-release convention and the chosen ONS ids, and let CI run.
  Do not merge.
