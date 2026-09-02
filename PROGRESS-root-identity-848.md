# PROGRESS — microcosm#848 root identity (branch `root-identity-848`)

Lane journal for hash-pinning every raw microdata input and referencing its
Chronicle registration. (The root `PROGRESS.md` belongs to the ACS
predictor-release-join lane; this file follows the `PROGRESS-<lane>.md`
convention already used by `PROGRESS-graph-acceptance.md`.) Historical once
merged — see CLAUDE.md "Root journals are history, not state".

## State

Started 2026-09-02 from origin/main `d2b31496`. Implementation complete;
verification and PR in progress. Nothing merged.

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

## Naming chosen (for chronicle#221)

Chronicle's live `db/data/*/manifest.yaml` uses a publisher-slug `source_id`
and a kebab-case, publisher-prefixed `package_id`
(`hmrc` / `hmrc-spi-income-bands-2023-24`), and `files[year]` holds exactly one
file per (package, year). So one package per distributed file:

| release | source_id | package_id | access |
| --- | --- | --- | --- |
| FRS 2024-25 (UKDS SN 9563), 14 tabs | `dwp` | `dwp-frs-2024-25-<tab>` | licensed |
| WAS round 8 household EUL (SN 7215) | `ons` | `ons-was-round-8-household` | restricted |
| LCFS 2023-24 household / person | `ons` | `ons-lcfs-2023-24-{household,person}` | restricted |
| ETB 1977-2024 household | `ons` | `ons-etb-1977-2024-household` | restricted |
| SPI Public Use Tape 2022-23 (SN 9422) | `hmrc` | `hmrc-spi-public-use-tape-2022-23` | restricted |
| CPS ASEC 2023 archive | `census_cps` | `census-cps-asec-2023` | public |
| SCF 2022 summary extract | `federal_reserve` | `federal-reserve-scf-2022-summary-extract` | public |

Note for chronicle#221: the FRS the UK build reads is **2024-25 / SN 9563**
(`tax_year_start: 2024`), not 2023-24, and the UK manifest carries **21**
licensed references over 14 distinct tabs.

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

- Full `uv run pytest` and `uv run ruff check .`, then push and open the PR.
