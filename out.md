# Pregnancy and prior-year defects: final report

Date: 2026-08-27

Branch: `pregnancy-prioryear-defects`

Lane base: `606cbd69` (`stacked-release-gate-alignment`)

## Outcome

Both assigned defects are resolved in code and covered by fail-closed
contracts: pregnancy has a source/transfer fix, while prior-year availability
has the owner-approved release-gate fix.

- Pregnancy is now structurally limited to female people ages 15 through 44
  before the ACS QRF draw. One source-person result is fanned across every
  missing assembled clone, missing ineligible values become deterministic
  false, and preexisting or final domain/clone violations are refused with
  explicit counts. Production transfer validation authenticates the structural
  receipt; the structural policy and execution contract are bound into
  checkpoint and target-bank identity.
- The prior-year availability shortfall is a sampling-order effect, not ACS
  dilution and not a transfer hole. Under the owner ruling, only the gate's
  applied lower availability floor is scaled by the authenticated production
  sampling rung. At rung 1.0 the gate and report follow the original path
  byte-for-byte. The authored `0.05` constant, the `0.50` upper bound, all
  other bands, thresholds, seeds, and batteries are unchanged.

Full repository verification is green: all five package shards passed in
independent pytest processes, repository Ruff passed, generated bundle and
coverage bytes are current, the CI test inventory passed, and the worktree has
no leftover graph artifact or untracked file.

This lane used the supplied 25% pool only as read-only diagnostic evidence. It
did not use the network, build or publish a pool/release, push, or publish any
artifact. The supplied pool still contains the old pregnancy outputs and is not
evidence that the source fix has run.

## Task 1: pregnancy structural eligibility

### Real-pool decomposition

The supplied pool is:

`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.h5`

It contains 1,970,973 person rows. The complete hard-domain decomposition is:

| Physical channel | Clone | Person rows | Pregnant nonfemale | Pregnant female outside 15--44 | Total violations |
|---|---:|---:|---:|---:|---:|
| ASEC | 0 | 108,073 | 0 | 0 | 0 |
| ASEC | 1 | 108,073 | 0 | 0 | 0 |
| ASEC | 2 | 17,987 | 0 | 0 | 0 |
| **ASEC all** |  | **234,133** | **0** | **0** | **0** |
| ACS | 0 | 856,626 | 45 | 30 | 75 |
| ACS | 1 | 856,626 | 61 | 28 | 89 |
| ACS | 2 | 23,588 | 2 | 0 | 2 |
| **ACS all** |  | **1,736,840** | **108** | **58** | **166** |
| **Pool** |  | **1,970,973** | **108** | **58** | **166** |

The initially reported row positions 129405, 167076, 171133, 192443, and
195546 are members of the 108-row nonfemale subset. All 166 hard-domain
violations occur on ACS physical records and are isolated to one clone within
their source-person group. There are 11,287 ACS source people whose assembled
clones disagree on pregnancy; ASEC has zero clone disagreements. Sex values,
ages, physical channel, and clone attachment are internally consistent, which
rules out assembly or sex-code corruption.

### Root cause

The defect was the ACS pregnancy QRF path. The ASEC source producer already
conditions its stable pregnancy draw on female ages 15 through 44
(`packages/microcosm-build/src/microcosm/build/us_runtime/pregnancy.py:292` and
`:330`). ACS transfer instead used sex and age only as soft QRF predictors,
modeled physical clone rows separately, and had neither a hard eligibility
precondition nor a domain/clone postcondition. That combination explains both
the 166 impossible positives and the 11,287 source-person disagreements.

### Source fix and construction invariant

The policy is explicit and identity-bearing at
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:141`:

1. Female status must be an exact `0/1` Boolean-domain value and age must be
   complete, finite, and within the inclusive range 15--44. Donor and recipient
   surfaces are checked before any requested pregnancy QRF runs
   (`acs_transfer.py:973`).
2. Assembled recipients are grouped by `person_source_id`, must have valid
   clone indices and exactly one clone 0, and must agree on eligibility and any
   preexisting pregnancy value (`acs_transfer.py:1025` and `:1100`).
3. Only one eligible, unresolved clone-0 representative per source person is
   sent to the QRF. Existing valid source-person values are preserved; missing
   ineligible rows are resolved as structural false without entering the QRF
   (`acs_transfer.py:1181`).
4. The one decoded source-person result is fanned to every missing clone
   (`acs_transfer.py:1223`). Completeness, hard-domain validity, and all-clone
   equality are then enforced as fail-closed postconditions
   (`acs_transfer.py:1248`).
5. Pregnancy preflight still runs when its surface is already complete or when
   some other requested family remains active. A complete surface therefore
   carries a zero-imputation structural proof instead of bypassing validation
   (`acs_transfer.py:1417`). Pregnancy is also isolated from unrelated bounded
   QRF families (`acs_transfer.py:3157`).

The ASEC source-stage producer and the release pregnancy signal gate apply
the same hard domain and exact-Boolean contract. Their details report the
actual domain and clone counts, and they refuse violations rather than
silently zeroing them
(`packages/microcosm-build/src/microcosm/build/us_runtime/pregnancy.py:167`,
`:224`, `:265`, and `:438`).

### Receipt and execution identity

The sealed structural receipt distinguishes four disjoint fill categories:
QRF representative rows, QRF clone fanout, preexisting-value fanout, and
ineligible-false assignments. It also records source-person topology and the
preexisting/final domain and clone-violation counts
(`acs_transfer.py:1189` and `:1285`). A successful new transfer must receipt
zero violations; a rejected surface names the nonzero count in its failure.

Production validation authenticates the policy digest, exact integer count
types, zero-violation postconditions, source-person/clone arithmetic, one QRF
row per QRF source person, and exact equality between the four fill categories
and the transferred-row count
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4203`
and `:4873`). Receipt emission is at `stacked_spine.py:10591`.

The structural policy is inside the target-specific transfer execution
contract and its SHA (`acs_transfer.py:276`). Each late-transfer model binding
includes that contract (`stacked_spine.py:6691`), the outer checkpoint identity
authenticates those resource semantics (`stacked_spine.py:6809` and
`tools/build_us_multispine_pool.py:1410`), and late target-bank identities
extend the checkpoint identity (`tools/build_us_multispine_pool.py:3300` and
`:3726`). A policy change therefore invalidates both checkpoint reuse and
target-bank reuse.

The generated US imputation authority declares the generic policy template at
`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:412` with
`enabled: false`; the resolver enables and rehashes it exactly for transfer
groups containing `is_pregnant`
(`packages/microcosm-build/src/microcosm/build/spec_engine/imputation_semantics.py:195`).
The late producer also declares `person_source_id` as its structural grouping
input (`packages/microcosm-build/src/microcosm/build/us_runtime/us_late_producer_registry.py:670`).
The final resolved US spec SHA is
`11e310c7619cbac91f6703b9679649cdd15f6fb09274ad29904c65881aa93316`.

### Pregnancy regressions

- Source refusal, gate counts/clone disagreement, and near-Boolean refusal:
  `packages/microcosm-build/tests/test_us_pregnancy.py:251`, `:302`, and `:344`.
- One draw per source person, clone fanout, structural false, and disjoint
  receipt accounting: `packages/microcosm-build/tests/test_us_acs_transfer.py:1448`.
- Donor/recipient pre-QRF and mixed-active refusal: `test_us_acs_transfer.py:1592`.
- Zero-imputation receipt and policy-dependent execution identity:
  `test_us_acs_transfer.py:1694` and `:1722`.
- Integrated stacked transfer, receipt propagation, all-clone equality,
  complete-surface proof, and invalid source refusal:
  `packages/microcosm-build/tests/test_us_stacked_spine.py:6311`.
- Bound execution-contract emission and stale-contract refusal:
  `test_us_stacked_spine.py:4651` and `:4693`.
- Forged policy-digest rejection:
  `packages/microcosm-build/tests/test_us_multispine_pool_tool.py:3026`.

## Task 2: prior-year availability

### Real-pool decomposition and verdict

| Physical channel | Clone | Person rows | Available rows | Weighted availability |
|---|---:|---:|---:|---:|
| ASEC | 0 | 108,073 | 4,724 | 0.04308325 |
| ASEC | 1 | 108,073 | 4,724 | 0.04308351 |
| ASEC | 2 | 17,987 | 752 | 0.04277737 |
| **ASEC all** |  | **234,133** | **10,200** | **0.04308325** |
| ACS | 0 | 856,626 | 38,228 | 0.04258969 |
| ACS | 1 | 856,626 | 38,228 | 0.04258837 |
| ACS | 2 | 23,588 | 928 | 0.04374427 |
| **ACS all** |  | **1,736,840** | **77,384** | **0.04258969** |
| **Pool** |  | **1,970,973** | **87,584** | **0.04283879** |

This is neither ACS dilution nor an ACS transfer hole. ASEC and ACS are both
about 4.3% available. ACS is intended to receive
`previous_year_income_available`: the generated authority declares the early
ASEC-to-ACS QRF over all recipient rows at
`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:1077`.
The nearly identical ASEC and ACS incidence is evidence that transfer is
operating.

The cause is the assembly's sampling order. ASEC households are sampled before,
and without coupling to, their adjacent-year `PERIDNUM` partners
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:633`
and `prior_year_income.py:346`). Of 18,518 sampled current-year records that
match the intact full predecessor files, only 4,724 retain a predecessor after
the sampled-to-sampled join. Weighted match survival is 25.4117%, the expected
25% rung effect. As controls, full pooled ASEC availability is 16.9147%, while
selected current rows joined to intact prior files are 16.9541%.

The owner accepted this sampling-order verdict. It falls under the charter's
"something else" category: the transfer is sound, and the low incidence is
the expected consequence of independently sampling both join sides.

### Rung-aware release floor

The authored bands remain exactly:

- `_PREVIOUS_YEAR_AVAILABLE_SHARE_BAND = (0.05, 0.50)`
- `_SELF_EMPLOYMENT_NONZERO_SHARE_BAND = (0.01, 0.25)`

They are at
`packages/microcosm-build/src/microcosm/build/us_runtime/prior_year_income.py:183`.
For a production sampling rung `r`, only the applied lower availability floor
becomes `0.05 * r`; the upper stays `0.50`.

| Rung | Applied availability floor | Upper bound |
|---:|---:|---:|
| 0.01 | 0.0005 | 0.50 |
| 0.04 | 0.0020 | 0.50 |
| 0.10 | 0.0050 | 0.50 |
| 0.25 | 0.0125 | 0.50 |
| 1.00 | 0.0500 | 0.50 |

The supplied candidate is gate-failed and contains stale pregnancy output; it
is not simulation-ready. Its sibling manifest nevertheless records and
digest-binds a valid sampling receipt at rung `0.25` (version 4, seed 578), so
the prior-year gate's applied floor is `0.0125`. The configured rung, not the
empirical 25.4117% diagnostic estimate, is the execution input.

On sampled production assemblies, the gate adds these report fields so the
reader can see exactly why the floor moved:

- `previous_year_income_available_sampled_match_survival_factor`
- `previous_year_income_available_applied_floor`
- `previous_year_income_available_applied_share_band`

The conditional logic is at `prior_year_income.py:889`. At rung 1.0 the branch
is not entered: no new fields are emitted, the original band key is evaluated,
and the result is byte-identical to the pre-change gate. No-manifest frames and
legacy version-1 frames also retain factor 1.0. Malformed version-4 metadata or
unsupported manifest versions fail closed (`prior_year_income.py:189`). No
other prior-year check or band changed.

### Manifest provenance and regressions

Production assembly writes the version-4 common fraction/seed and both arm
receipts into frame metadata (`stacked_spine.py:679`). Pool publication carries
the summarized sampling receipt, full stack manifest, and canonical SHA-256
binding (`tools/build_us_multispine_pool.py:4193` and `:4224`). The H5 loader
checks the schema-9 envelope, approved rung token, strict finite-float fraction,
non-Boolean seeds/counts, exact ASEC+ACS arms, arm/top agreement, positive
realized counts, and canonical manifest digest before restoring a deep copy to
frame metadata. The surrounding loader also authenticates the H5 SHA, size, and
run identity and rechecks the digest after reading
(`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:238`, `:270`,
`:819`, `:1750`, and `:1813`). On the production release path the gate therefore
consumes loader-authenticated, not caller-invented, rung metadata
(`tools/build_us_fiscal_refresh_release.py:1696` and `:9514`).

Regression coverage includes:

- scaled-only lower-floor behavior and unchanged bands:
  `packages/microcosm-build/tests/test_us_prior_year_income.py:439`;
- lower and upper enforcement: `test_us_prior_year_income.py:468`;
- exact rung-1 gate/report bytes: `test_us_prior_year_income.py:486`;
- legacy behavior and malformed metadata: `test_us_prior_year_income.py:510`;
- receipt restoration and top/arm disagreement:
  `packages/microcosm-build/tests/test_us_multispine_pool_h5_io.py:1456`;
- nested Boolean/type-alias refusal: `test_us_multispine_pool_h5_io.py:1537`.

Pre-sampling adjacent-year joins at assembly remain a possible architectural
end state only. This lane did not move the joins or change source assembly.

## Diff summary

Before replacing this report, the lane diff comprised 34 files with 2,352
insertions and 126 deletions. The substantive changes are:

- pregnancy source, ACS transfer, stacked receipt/authentication, late-producer
  structural inputs, execution identity, and their source/transfer/gate/tool
  regressions;
- authenticated H5 sampling-receipt restoration and the rung-aware prior-year
  gate/report regressions;
- generated US imputation authority, schema/semantic projection, compiler
  field-usage and inventory proofs, and the source-attested BE/UK/minimal hash
  repins caused by the shared semantic change;
- coverage evidence and the additive optional-ACS provenance golden; and
- changelog fragments
  `changelog.d/798-pregnancy-structural-transfer.fixed.md` and
  `changelog.d/799-prior-year-rung-floor.fixed.md`.

## Verification

All commands used the prebuilt environment with no sync or network access.

### Complete independent pytest shards

| Shard target | Result |
|---|---:|
| `pytest packages/microcosm-build/tests` | 6,601 passed, 45 skipped |
| `pytest packages/microcosm-calibrate/tests` | 203 passed |
| `pytest packages/microcosm-data/tests` | 318 passed, 2 skipped |
| `pytest packages/microcosm-fit/tests` | 93 passed |
| `pytest packages/microcosm-frame/tests` | 295 passed, 36 skipped |
| **Aggregate** | **7,510 passed, 83 skipped** |

The clean final build-shard process took 55m20s and exited 0. Its 2,351
warnings are preexisting overflow, pandas chained-assignment/fragmentation,
PolicyEngine division, and donor-snap warnings; there were no test errors.

Focused evidence also passed: all 23 prior-year tests and all 64 current H5
loader tests (the initial 60-test run plus four later nested-type adversarial
cases), the complete pregnancy and ACS-transfer files, integrated stacked
execution, forged-policy controls, the complete optional-ACS multispine file,
and the country-bundle/loader/spec matrix.

### Static, generated, and workspace gates

- `uv run --no-sync ruff check .`: `All checks passed!`
- `uv run --no-sync python tools/ci_test_groups.py --verify`:
  `tracked_test_files=309`, `verification=ok`
- `uv run --no-sync python tools/generate_us_bundle_from_constants.py --check`:
  `US bundle spec_sha256=11e310c7619cbac91f6703b9679649cdd15f6fb09274ad29904c65881aa93316`
- `uv run --no-sync python tools/spec_engine_coverage.py --check`:
  `42154/42154` configuration fields and `41/41` inventory checks
- `git diff --check 606cbd69..HEAD`: passed
- Untracked files and local `.gitnexus` directories: zero

## Judgment calls and boundaries

- Missing values on ineligible recipients are constructed as false and
  receipted; preexisting invalid true values are refused with counts. This
  preserves valid existing data without silently repairing corrupted input.
- One eligible clone-0 representative owns the random draw. Every missing clone
  of the source person receives the same result, while valid existing values
  are preserved, eliminating clone-specific random pregnancy states by
  construction.
- Boolean structure is exact `0/1`; near-Boolean floats do not pass through
  numerical tolerance. Nested manifest fractions, seeds, and counts likewise
  reject JSON Boolean equality aliases.
- A complete pregnancy surface cannot bypass structural preflight during a
  restart and carries a zero-imputation proof.
- The prior-year gate uses the authenticated configured rung as the
  sampled-to-sampled survival factor. It does not substitute a noisy empirical
  estimate, change the authored 0.05 constant, relax the upper bound, or touch
  another band.
- Legacy/no-rung and full-rung behavior remains unchanged. At rung 1.0 the
  gate report is byte-identical to the old output.
- Joining adjacent raw ASEC years before sampling is documented only as a
  future architectural end state, per the ruling; it is not implemented here.

## Commit inventory before this report carrier

1. `b7be9c35` — Start pregnancy and prior-year defect journal
2. `364f4525` — Record pregnancy and prior-year root causes
3. `6d3351fd` — Record prior-year release-gate ruling
4. `f5284a07` — Scale prior-year availability gate by sampling rung
5. `4aa6269a` — Record prior-year gate implementation
6. `01a80f49` — Enforce structural pregnancy transfer eligibility
7. `e59ab046` — Harden pregnancy and rung receipts
8. `d2b75e1e` — Repin shared spec attestations
9. `7caf69ac` — Repin ACS transfer provenance fixture
10. `011563db` — Record completed provenance repin
11. `c1a4bfb9` — Record full shard verification

## Host-owned next action

Rebuild the affected pregnancy transfer from the invalidated checkpoint and
target bank, then run the terminal pool/release gates on the rebuilt artifact.
The prior-year gate will read the existing authenticated sampling receipt and
report the applied factor/floor. Those build, certification, publication, and
push actions are deliberately outside this headless lane.
