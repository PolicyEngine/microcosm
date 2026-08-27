# ACSPRED: release-side ACS predictor join final report

Date: 2026-08-27

Branch: `acs-predictor-release-join`

Lane base: `606cbd69` (`stacked-release-gate-alignment`)

## Outcome

The release builder now populates every CPS-named predictor consumed by the six
archived donor models on every physical ACS support row. It does so through an
exact release-time join to the two canonical, SHA-pinned 2024 one-year ACS PUMS
archives and a versioned, digest-pinned crosswalk. It does not fill unknowns
with silent defaults, weaken a gate, retrain a model, or change model selection
logic.

The join passed read-only on the complete supplied candidate: 856,626 unique
ACS source people matched exactly 856,626 raw person records in 382,903 raw
households and fanned out to 1,736,840 ACS support rows. All model predictors
have zero remaining nulls. The SSI reporter path separately preserves 261,605
genuine below-age-15 ACS support-row nulls and treats them as false only at the
consumer's `> 0` predicate.

The implementation adds four all-or-none release CLI inputs, writes the full
receipt to both manifests, keeps legacy non-ACS releases as an explicit
disabled identity path, and leaves the launcher contract to the dispatcher as
required.

No network access, pool build, release build, wheel build, publication, push,
or launcher edit was performed.

## Join-key verification

### Why `person_source_id` is not the raw key

The ACS loader rejects duplicate `(SERIALNO, SPORDER)` pairs, stably sorts by
that pair, and only then assigns a zero-based raw `source_row_id`
(`acs_pums.py:204-238`). Source construction stores `SPORDER` separately as
`source_person_id` and links people to the household identified by `SERIALNO`
(`acs_pums.py:264-279`). Assembly then collision-offsets structural IDs and
creates assembly-unique support source IDs (`spine_assembly.py:366-430`;
`support_provenance.py:346-371`). Therefore neither the final structural ID nor
`person_source_id` is a reversible Census record key.

The release join instead recovers the retained semantic identity:

| Role | Retained evidence | Enforced relation |
|---|---|---|
| Household key | linked household `SERIALNO` | nonblank, one source household identity per serial |
| Person key | integral `SPORDER` and `source_person_id` | exactly equal and at least 1 |
| Raw ordinal | `source_row_id`, `person_spine_source_id` | exactly equal, but never used as the Census join key |
| Source identity | `person_source_id` | one raw identity across all clones; fan-out only |
| Clone identity | `person_support_clone_index` | unique `(person_source_id, clone_index)` and exactly one clone 0 |
| Vintage | `source_year` | exactly 2024 on physical ACS rows |

The implementation derives these relations at
`acs_release_predictors.py:1078-1289`, including person-to-household channel
agreement, clone agreement, raw ordinal agreement, and unique canonical
`(SERIALNO, SPORDER)`. It then performs a pandas `one_to_one` left merge on
that semantic key, rejects any unmatched pool source person, and refuses a row
count change (`acs_release_predictors.py:893-923`). Only after that proof does
it map values back through `person_source_id` to every clone
(`acs_release_predictors.py:939-968`).

## Artifact and refusal contract

The reviewed archive identities are declared at
`acs_release_predictors.py:70-88`:

| Archive | SHA-256 | Size observed |
|---|---|---:|
| 2024 ACS person `csv_pus.zip` | `afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894` | 602,847,146 bytes |
| 2024 ACS household `csv_hus.zip` | `8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0` | 251,500,587 bytes |

Before either zip is opened, the boundary requires lowercase 64-hex syntax,
requires the caller's expected hash to equal the reviewed canonical pin, and
hashes the actual file (`acs_release_predictors.py:991-1021`). It then requires
exact `psam_pusa/b.csv` or `psam_husa/b.csv` membership, verifies required
headers, and streams only selected serials (`acs_release_predictors.py:1024-1075`).

The boundary also refuses:

- partial CLI/archive inputs or archive inputs on a frame with no ACS rows;
- a stale crosswalk digest;
- malformed or incomplete assembly provenance;
- duplicate raw person or household keys;
- source-identity, semantic-key, or clone-index collisions;
- orphaned or cross-channel person/household links;
- pool household `TEN` disagreement with the pinned household archive;
- an incomplete one-to-one person join;
- an unsupported raw code or age/universe violation;
- a conflicting pre-existing ACS predictor value;
- missing, nonnumeric, or nonfinite native ASEC predictors, or negative
  `SSI_VAL`; and
- any remaining null in a model-consumed predictor.

The relevant executable checks are at
`acs_release_predictors.py:830-968,1078-1356,1548-1586`.

## Reviewed crosswalks

The canonical crosswalk payload is source-controlled at
`acs_release_predictors.py:726-790` and pinned as:

`1d4906242e9c73e31b3283659e5cad8242b8cbc42914ab6fa59547a10c8770e9`

| Source | Target | Exact reviewed mapping | Consumed model bin |
|---|---|---|---|
| `DDRS` | `PEDISDRS` | age 5+: `1 -> 1`, `2 -> 2`; younger blank `-> -1` | SSI tests only `== 1` |
| `DEAR` | `PEDISEAR` | all ages: `1 -> 1`, `2 -> 2` | SSI tests only `== 1` |
| `DEYE` | `PEDISEYE` | all ages: `1 -> 1`, `2 -> 2` | SSI tests only `== 1` |
| `DOUT` | `PEDISOUT` | age 15+: `1 -> 1`, `2 -> 2`; younger blank `-> -1` | SSI tests only `== 1` |
| `DPHY` | `PEDISPHY` | age 5+: `1 -> 1`, `2 -> 2`; younger blank `-> -1` | SSI tests only `== 1` |
| `DREM` | `PEDISREM` | age 5+: `1 -> 1`, `2 -> 2`; younger blank `-> -1` | SSI tests only `== 1` |
| `RAC1P` | `PRDTRACE` | `1 -> 1` White; `2 -> 2` Black; `6 -> 4` Asian; `3/4/5/7/8/9 -> 3` residual Other | SCF White/Black/Asian/Other; ORG White/Black/Other |
| `HISP` | `PRDTHSP` | `1 -> 0` non-Hispanic; `2..24 -> 1` positive Hispanic representative | both consumers test zero versus positive |
| observed `OCCP` | `PEIOOCC` | identity over all 530 pinned detailed codes | SIPP tips exact detailed-code membership |
| blank `OCCP` | `PEIOOCC` | `-1`, the CPS NIU sentinel | unlisted/non-tipped |
| observed `OCCP` | `POCCU2` | explicit 530-key table covering consumed bins 1 through 53 | ORG exact categories and EAP set |
| blank `OCCP` | `POCCU2` | age below 16 `-> 0`; age 16+ permitted only with `ESR=6`, then `-> 53` | preserve NIU; adult no occupation/never worked |
| household `TEN` | `SPM_TENMORTSTATUS` | `1 -> 1`, `2 -> 2`, `3/4 -> 3`; verified group-quarters blank `-> 3` | SIPP vehicle homeowner iff `{1,2}` |

The disability, race, occupation, and tenure executable mappings are at
`acs_release_predictors.py:108-687,689-694,1364-1454`. The occupation map's
notable reviewed edges include `3250 -> 26` while `3255/3256/3258 -> 25`, ACS
military `9800/9810/9825/9830 -> 52`, and `9920 -> 53`. The table maps the
shared detailed code vocabulary to the modal 2024 ASEC category; it is not
represented as a fictional rowwise CPS identity.

### Age-15 occupation judgment

ACS `OCCP` begins at age 16, while CPS `POCCU2` is already in universe at age
15. In source-year-2024 ASEC, 2,174 people are age 15: 1,931 have `POCCU2=53`
but 243 have categories 1 through 52. Mapping every ACS age-15 blank to 53
would therefore invent never-worked evidence. The crosswalk preserves 0 for
that one-year source/target universe gap and uses 53 only for defensible adult
blanks. Raw `ESR` is required blank below 16 and in `1..6` from 16; an adult
`OCCP` blank is allowed only at `ESR=6`.

## SSI reporter semantics

`SSI_VAL` is the measured ASEC amount. The pool's native ACS mapping already
stores adjusted `SSIP` as `ssi_reported`. The release join now reads raw `SSIP`
and `ADJINC`, enforces the exact age-15 universe, and proves clone-zero
`ssi_reported == SSIP * ADJINC / 1_000_000` source person by source person
(`acs_release_predictors.py:1458-1545`).

The SSI receiver coalesces `SSI_VAL` and `ssi_reported` rowwise, without
origin-routing the model, rejects conflicting dual reporter statuses, permits
a blank only below age 15, and fills that blank only transiently for the
existing `> 0` predicate (`ssi_disability_criteria.py:758-813,1082-1093`). The
gate summary uses the same coalesced anchor and native-role scope, so a lost
positive ACS-native reporter cannot evade the diagnostic. Source nulls remain
null in the frame.

## Release integration and receipts

The release builder declares the four inputs at
`build_us_fiscal_refresh_release.py:1293-1318` and validates all-or-none plus
lowercase SHA syntax at `:1587-1610`:

- `--acs-person-zip`
- `--acs-person-sha256`
- `--acs-household-zip`
- `--acs-household-sha256`

It invokes the join at `build_us_fiscal_refresh_release.py:9828-9846`, before
SCF wealth and consequently before all six archived-model stages. The receipt
is embedded at top level in `build_manifest.json` (`:7534-7642`) and under
`release_manifest.json.build` (`:7841-7871`); the sole writer call receives the
saved runtime receipt at `:11860-11879`.

The complete real-data receipt included this core evidence:

```json
{
  "enabled": true,
  "crosswalk": {
    "version": 1,
    "sha256": "1d4906242e9c73e31b3283659e5cad8242b8cbc42914ab6fa59547a10c8770e9"
  },
  "join": {
    "semantic_key": ["household.SERIALNO", "person.SPORDER"],
    "clone_fanout_key": "person_source_id",
    "acs_source_people": 856626,
    "acs_support_rows": 1736840,
    "acs_support_rows_by_clone_index": {"0": 856626, "1": 856626, "2": 23588},
    "selected_raw_person_rows": 856626,
    "selected_raw_household_rows": 382903,
    "unmatched_pool_source_people": 0,
    "source_identity_collisions": 0,
    "semantic_key_sha256": "c7723adc889fc655b46103426d21f6c74937434eff51e8f3e6025bdf60972b74"
  }
}
```

Per-model counts were:

| Model | Predictors | ASEC-native each | ACS-joined each | Still null |
|---|---|---:|---:|---:|
| SSI disability | six `PEDIS*` | 234,133 | 1,736,840 | 0 |
| SCF wealth | `PRDTRACE`, `PRDTHSP` | 234,133 | 1,736,840 | 0 |
| SCF auto loans | `PRDTRACE`, `PRDTHSP` | 234,133 | 1,736,840 | 0 |
| SIPP vehicles | `SPM_TENMORTSTATUS` | 234,133 | 1,736,840 | 0 |
| SIPP tips | `PEIOOCC` | 234,133 | 1,736,840 | 0 |
| ORG wages/FLSA | `PRDTRACE`, `PRDTHSP`, `POCCU2` | 234,133 | 1,736,840 | 0 |
| SSI logical reporter anchor | ASEC `SSI_VAL`; ACS `ssi_reported` | 234,133 | 1,475,235 | 261,605 child-universe nulls |

The receipt producer is at `acs_release_predictors.py:1589-1677`.

## Archived-model behavior retained

The five non-SSI consumer modules are byte-unchanged from the lane base. The
SSI model's predictor list, QRF feature/selection logic, role routing, and
thresholds are unchanged; only its source-faithful reporter read and matching
gate diagnostic changed.

| Consumer | Executable consumed bins | Evidence |
|---|---|---|
| SSI disability | six difficulty inputs consumed only as `== 1`; reporter is under 65 and `> 0` | `ssi_disability_criteria.py:143-150,902-908,1082-1093` |
| SCF wealth | White 1, Black 2, Asian 4, Hispanic positive, residual Other | `scf_wealth.py:611-621,654-724` |
| SCF auto loans | reuses the SCF wealth race helper | `scf_auto_loans.py:38-42,365-380` |
| SIPP vehicles | homeowner iff tenure code is 1 or 2 | `sipp_vehicles.py:577-621,740-763` |
| SIPP tips | exact detailed Census occupation membership; unlisted NIU is zero | `sipp_tips.py:129-184,249-263,363-375` |
| ORG wages/FLSA | POCCU2 53 never worked, 52 military, 8 computer, 41 farmer/fisher, explicit EAP set; Hispanic/White/Black/Other | `org_wages.py:133-183,408-415,535-559,610-625` |

No gate band or numeric threshold was edited. Whether ORG and tips pass or fail
on a future release is therefore determined by the real joined values, as
required.

## Real-pool audit and operational caveat

The supplied candidate pool contains 1,970,973 people and 865,460 households.
Its 3,239,263,147-byte H5 exactly matches the frozen manifest SHA-256
`871b7e6467675a1e9475b54fd1baf64c53c0f75a3258b8357303a8df0d53642d`.

The current official release loader refuses that older candidate before H5
loading because the manifest's archived primary-QRF worker binding predates
this branch's source-attested execution identity: `late primary-QRF worker
binding changed`. To isolate and test this lane without building an artifact,
the audit independently verified the H5 bytes, read its entity tables through
the repository HDF reader, restored the frozen `assembly_receipt`, and ran the
strict join. Every join and crosswalk assertion passed.

This is not an unresolved crosswalk issue, but it is an operational handoff:
the old candidate cannot be promoted through the current authenticated release
loader. A fresh/currently authenticated candidate is required outside this
headless no-build lane.

## Verification

All commands ran offline against the prebuilt environment. Each package shard
ran in its own pytest process. Packaging was not touched, so wheels were not
built.

- `uv run --no-sync ruff check .`: PASS, `All checks passed!`
- `uv run --no-sync python tools/ci_test_groups.py --verify`: PASS,
  `tracked_test_files=310`, `verification=ok`
- `tools/generate_us_bundle_from_constants.py --check`: PASS, US spec
  `16b7d5e622e8a68e008165bb44a5836695d94a2b0dd8d4c51b3c9e8ca89dca38`
- `tools/spec_engine_coverage.py --check`: PASS, 42,122/42,122 fields and
  41/41 inventory checks
- frame shard: PASS, 295 passed, 36 skipped
- fit shard: PASS, 93 passed
- calibrate shard: PASS, 203 passed
- data shard: PASS, 318 passed, 2 skipped
- build shard after reviewed source-attestation repin: PASS, 6,586 passed,
  45 skipped
- join-focused file: PASS, 15 passed
- combined join, SSI, and source-blindness files: PASS
- parser/order/manifest focused cases and all six parametrized mocked-main
  corridor cases: PASS
- repository `git diff --check`: PASS

The first complete build-shard run reached 100% with 6,575 passed and 45
skipped plus five failures and six setup errors, all caused by stale expected
source-attestation hashes after changing `ssi_disability_criteria.py`. That
module participates in both seed-kernel source inventories. The narrow
established repin updated the two seed digests, BE/UK/US spec identities, the
minimal-loader golden, and generated coverage evidence. All 25 affected tests
then passed, both retained `--check` commands passed, and the complete build
shard rerun above was green.

Existing warnings were numerical overflow/divide, pandas chained-assignment,
and fragmented-fixture performance warnings; none was a test failure.

## Judgment calls and unresolved items

- The join uses retained semantic lineage, not a guessed arithmetic inversion
  of assembly IDs.
- Race and Hispanic mappings stop at the bins the consumers actually read;
  they do not invent CPS detail absent from ACS.
- Disability codes preserve `1/2` plus the CPS NIU sentinel rather than
  collapsing the source to booleans.
- Age-15 ACS occupation remains NIU because assigning 53 would fabricate
  never-worked evidence for a group whose CPS distribution is not degenerate.
- `PEIOOCC` blank uses CPS NIU `-1`, not a fictional detailed occupation 0.
- SSI child nulls remain source nulls; only the predicate interprets them as
  non-reporters.
- Pool household `TEN` is preferred for lineage and must agree exactly with the
  pinned household zip; group-quarters non-owner status is applied only after
  raw universe checks.
- ASEC values are validated but never rewritten, preserving their native model
  inputs byte-for-byte.
- No crosswalk semantic remains unresolved. The only open operational item is
  obtaining a current authenticated pool and adding the four launcher
  arguments; both are explicitly outside this lane and owned by the
  dispatcher/build process.

## Commits before final journal handoff

- `2aa14e84` Start ACS predictor release join journal
- `c1a41ccf` Record ACS predictor join contracts
- `ecea55a2` Add strict ACS release predictor join
- `a7108697` Harden ACS predictor crosswalk contracts
- `6b8185e4` Wire ACS predictor join into release builder
- `d87be068` Accept numeric H5 predictor cells
- `69ec0fae` Document ACS release predictor join
- `de5e5d03` Repin source-attested spec identities

The final local commit adds this report and marks `PROGRESS.md` complete.
