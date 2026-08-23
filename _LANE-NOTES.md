# Battery residual-fixes lane notes

## Residual lane boundary — 2026-08-22 15:50Z

- Starting tracked revision: `bb94f789` on `battery-residual-fixes`, stacked
  on `battery-pkg3-two-part`.
- Owned checks are limited to adult-care post-reconciliation, model-required
  targeted calibration, Schedule D joint-parent reconciliation, and any
  still-red post-pkg3 SSI or weeks-unemployed incidence checks. PUF tax, QBI,
  rare tails, health labels, retirement, exclusions, and all gate policy are
  owned elsewhere.
- This lane will not run a pool build. It will read the baseline and pkg3 gate
  artifacts, create code/tests/journals only on this branch, and hand exact
  checkpoint-based 1% rerun commands to the serial host owner.
- Normal GitNexus graph queries are unavailable. Mechanisms will be established
  by direct adjudication, source, call-site, history, regression, and read-only
  checkpoint tracing, with module-and-line citations in the final record.
- The mandated US-extra sync was attempted before repository edits. Sandbox
  cache and network restrictions prevented resolution; the complete pkg3
  environment was copied locally and its five editable shards were repointed
  to this worktree for `uv run --no-sync` verification.
- Earlier pkg3 notes below are retained as source-cited history and do not
  describe this lane's current state or authority.

## Post-pkg3 residual adjudication — 2026-08-22

### Frozen evidence and mirror-deduplicated red universe

- Read-only baseline gate SHA-256:
  `1d6059868680f872fe04d452a536bcc3c215bafabb4c50d7740a469fe6a8b56a`.
  Read-only pkg3 gate SHA-256:
  `3ace0af0fd9e2ed6cb37cb110280f0c5cade182118c62737635c7ad177050ac3`.
- The canonical
  `.terminal_gates.gates.us_by_origin_battery.failures` arrays contain 127
  unique baseline failures and 114 unique pkg3 failures. In pkg3,
  `.terminal_gates == .agreement_gate`; counting one canonical copy rather
  than both identical mirrors is the required mirror de-duplication. Artifact
  schema 8 reports a complete `us-stacked-pool` terminal-gate evaluation.
- This table is the named residual slice of those 114 checks. It was computed
  before choosing a code change; no comparator or gate input was modified.

| Check at clone 0 | pkg3 result | Lane verdict |
| --- | --- | --- |
| Adult-care expense positive incidence | `1.024210809` (`4` ASEC / `27` ACS positive rows) | Assigned check green. |
| Adult-care expense positive QED | `leg_insufficient_support` because ASEC has four positives | Assigned mechanism ran; frozen five-row QED support rule correctly emits no value. |
| Adult-care incapacity flag incidence | `0.548455830` (`72` / `414`) | Still red, but a distinct boolean QRF target rather than either assigned expense check. |
| Unemployment-compensation positive QED | `0.0` (`34` / `32`) | Assigned check green; move on as directed. |
| Unemployment-compensation positive incidence | `0.131859569` | Distinct unassigned carrier criterion; the assigned remedy freezes carriers. |
| Schedule D distribution positive incidence | `1.092661185` (`119` / `1,263`) | Assigned joint-parent check green. |
| Schedule D distribution positive QED | `0.352941176` | Different check inherited from still-red PUF-tax parent amounts. |
| Simulated SSI positive incidence | `1.335482545` (`92` / `1,183`) | Still red; upstream formula-input diagnosis below. |
| Weeks-unemployed positive incidence | `0.031371146` (`134` / `32`) | Still red and locally repairable. |
| Weeks-unemployed positive QED | `0.0` | Green under pkg3; only incidence changes here. |

### Adult-care expense checks: already resolved by pkg3

- Mechanism: person-grain QRF output can place expense on a nonqualifying or
  duplicate person. The reconciliation defines the section-21 qualifying mask,
  clears only mutable ineligible/duplicate carriers, lets an immutable positive
  block additions in its tax unit, and preserves immutable values
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:660-739`).
- Existing fix: before reconciliation, the late owner restricts carrier
  selection to qualifying mutable rows and at most one addition candidate per
  empty tax unit; the shared amount map writes only mutable positive recipient
  rows from positive ASEC donor support, including the frozen five quantile
  anchors (`stacked_spine.py:8698-8716,8930-8987`;
  `post_transfer_calibration.py:573-720`). It then requires reconciliation to
  be a byte-identical no-op (`stacked_spine.py:9019-9041`). The pkg3 execution
  receipt records 27 mapped rows, QED `1.738865343 -> 0`, preserved immutable
  bytes, and `verified_no_op` reconciliation.
- Regression: route qualification/reconciliation tests cover exclusivity and
  immutable blocking (`test_us_acs_transfer.py:2263-2333,2513-2574`); the kernel tests
  cover donor-support amount mapping and immutable bytes
  (`test_us_post_transfer_calibration.py:354-436`); terminal validation
  recomputes the live adult-care constraint
  (`test_us_multispine_pool_tool.py:2302-2341`).
- Expected/observed effect: the assigned incidence is green at `1.024210809`.
  At this 1% sample the amount mechanism ran and its receipt reports exact QED,
  but the independent battery correctly withholds QED because only four ASEC
  positives are below its fixed five-row support floor. No residual change is
  justified. The separate red incapacity flag remains an ordinary QRF target
  (`acs_transfer.py:337-344`) and is not altered to tune the expense checks.

### Unemployment-compensation amount check: QED

- Mechanism and existing fix: unemployment compensation is deliberately
  `preserve_recipient`, so its carrier membership stays frozen while the common
  positive-amount map rewrites only mutable positive amounts from ASEC positive
  donor support
  (`post_transfer_calibration.py:190-197,573-720,930-940,963-993`).
- Regression: the preserve-mode test requires identical carrier membership,
  all five exact quantile anchors, QED `0`, donor support, and the preserve-
  carriers receipt invariant
  (`packages/microcosm-build/tests/test_us_post_transfer_calibration.py:354-390`).
- Expected/observed effect: positive QED is exactly `0.0`; the sparse incidence
  ratio `0.131859569` is intentionally unchanged because it is not the assigned
  criterion. No new code is warranted.

### Schedule D joint-parent check: assigned incidence green

- Mechanism and existing fix: this memo leg is not independently fitted. It is
  deterministically derived as the packaged share of positive
  `long_term_capital_gains_before_response` only where the mutually exclusive
  `non_sch_d_capital_gains` route is absent
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:611-657`).
  The pool owner first sums both transferred parents at tax-unit grain and then
  puts the derived value on the first missing person while filling other
  missing members with zero, preserving observed rows
  (`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:2868-2946`).
- Regression: the derivation test requires the share and mutual-exclusion
  identities and rejects incomplete parents
  (`packages/microcosm-build/tests/test_us_acs_transfer.py:2461-2510`).
- Expected/observed effect: the adjudicated incidence is green at
  `1.092661185`. Its current QED `0.352941176` is not that adjudicated check;
  pkg3 still reports parent QED `1.2` for long-term gains and QED `1.34694`
  plus incidence `0.204467` for non-Schedule-D gains. A local Schedule D amount
  map would break its deterministic share identity and overlap the PUF-tax
  lane, so no residual change is made.

### SSI residual: upstream income-threshold mechanism, no local gate change

- The pkg3 take-up input is not the discriminator: its by-origin battery row is
  exactly `1.0 / 1.0`. The pool input owner fills unresolved non-seeded take-up
  cells with the installed engine default and records that provenance
  (`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:2970-3091`);
  the SSI contract declares default `true` and a separate SSI take-up owner
  (`packages/microcosm-build/src/microcosm/build/us/take_up_contract.json:121-145`).
  The pkg3 receipt confirms all 80,395 rows were defaulted, with zero seeded or
  preserved rows.
- PolicyEngine-US computes `ssi` by flooring `uncapped_ssi`, applying the
  spousal cap, and multiplying by take-up
  (`policyengine_us/variables/gov/ssa/ssi/ssi.py:13-40`). Eligibility combines
  aged/blind/disabled, resource, and immigration tests
  (`policyengine_us/variables/gov/ssa/ssi/is_ssi_eligible.py:10-18`), while
  `uncapped_ssi` subtracts countable income from the eligible amount
  (`policyengine_us/variables/gov/ssa/ssi/uncapped_ssi.py:11-16`). Countable
  income sends earned, unearned, parentally deemed, and in-kind support through
  the exclusions, then adds spousal deemed income afterward
  (`policyengine_us/variables/gov/ssa/ssi/eligibility/income/ssi_countable_income.py:28-87`;
  `policyengine_us/variables/gov/ssa/ssi/eligibility/income/_apply_ssi_exclusions.py:21-43`).
- Read-only formula decomposition on pkg3 reproduces the exact
  `1.3354825449451269` SSI incidence ratio. Weighted no-income-test eligibility
  is slightly *lower* on ACS (`0.18883741352468275`, 8,166 rows) than ASEC
  (`0.19703927204185412`, 787 rows). The divergence appears among those
  eligible when `uncapped_ssi > 0`: `0.14407772964965512` ACS versus
  `0.1033936552883328` ASEC. Eligible-person countable-income q10 is also lower
  on ACS (`6,884.259765625`) than ASEC (`8,880.0`). The decomposition used the
  same 5,000-household batched engine materialization as the pool and exactly
  reproduced the gate's SSI shares, identifying broad upstream earned/
  unearned/deemed-income shape rather than SSI take-up, an SSI-local carrier
  selector, or a battery band.
- Fix/regression verdict: SSI is formula-owned, is materialized only on an
  ephemeral gate view, and cannot be persisted
  (`multispine_pool.py:3094-3111`). No SSI row exists in the frozen
  adjudication authorizing a mutable surface over those upstream income
  distributions. Changing the take-up default or formula to force the output
  ratio would be an unauthorized scope/gate substitution. This lane records
  the diagnosis, adds no local-fix regression, and leaves the generating inputs
  to their owning packages; expected local effect is exactly none.

### Weeks-unemployed residual: implemented generating fix

- Mechanism: ASEC `LKWEEKS` is a direct integer `-1`/`0..52` carry independent
  of unemployment compensation (`packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py:791-830`).
  Zeroing weeks when unemployment compensation is nonpositive belongs only to
  the PUF QRF path (`weeks_unemployed.py:955-980`), and the signal audit checks
  that relationship only on `puf_mask` rows (`weeks_unemployed.py:1266-1276,1333-1337`).
  Pkg3 commit `33bf52fe` incorrectly promoted that PUF-only relationship into
  the later ASEC-to-ACS calibration
  (`33bf52fe:post_transfer_calibration.py:122-180,237-244`;
  `33bf52fe:stacked_spine.py:8863-8876`). Only 32 of 34,293 mutable ACS rows
  had positive unemployment compensation, so capacity saturation forced the
  ACS weeks share to the identical `0.0010719375` unemployment share and left
  the incidence ratio at `0.031371146`
  (`tools/reproduce_us_post_transfer_weeks_checkpoint.py:28-62,132-230,278-320`).
- Fix: the immutable policy now declares weeks as ordinary late
  `match_reference` with no special constraint
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:122-178,234-240`).
  The owner therefore supplies no restricted masks: all transferred-null then
  nonnull ACS clone-0 rows are mutable/allowed/addition candidates
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8930-8987`).
  Carrier selection still targets the weighted ASEC share and the amount map
  still uses only positive ASEC donor support while preserving all nonmutable
  bytes (`post_transfer_calibration.py:573-720,782-817,819-943,963-993`). Policy
  content hashing updates the authored lineage SHA
  (`post_transfer_calibration.py:297-344`;
  `test_imputation_lineage_spec.py:103-105`); no schema/version or gate changes
  are involved.
- Regression: `test_weeks_late_calibration_does_not_require_unemployment_carriers`
  supplies only zero-UC rows, requires positive integer donor-supported ACS
  weeks, default mutable masks, exact reference share, no capacity limit, and
  immutable-byte preservation
  (`packages/microcosm-build/tests/test_us_stacked_spine.py:6367-6438`). Policy
  identity and publication-validator tests cover the removed constraint and
  retain the adult-care-only live coupled constraint
  (`test_us_post_transfer_calibration.py:95-107`;
  `test_us_multispine_pool_tool.py:2302-2341`).
- Expected effect from the SHA-pinned no-build replay: allowed/addition rows
  rise `32 -> 34,293`; positive ACS rows rise `32 -> 2,174`; after-share is
  `0.0341923809` versus ASEC `0.0341695371`, for ratio `1.0006685424`.
  Capacity is not limited, all values remain integer donor support `1..46`,
  donor-support violations are zero, immutable bytes validate, and QED remains
  `0`. The harness pins the assembled, UC, and weeks artifacts and exposes both
  `legacy-positive-unemployment` and `current-mutable` scopes
  (`tools/reproduce_us_post_transfer_weeks_checkpoint.py:76-84,91-129,132-320,323-378`).

The exact read-only replay command is:

```bash
UV_CACHE_DIR=/private/tmp/microcosm-residual-uv-cache \
  uv run --no-sync python tools/reproduce_us_post_transfer_weeks_checkpoint.py \
  --checkpoint-stage-root /Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/pkg3/pool.checkpoints/stacked/8f5077d6a1d5440b241f22fe4d20ad1d889924a27d094cb669e1035f9306546b \
  --carrier-scope current-mutable \
  --expect valid
```

### Final local verification

- Ran each required package test root separately on the complete implementation
  tree with `UV_CACHE_DIR=/private/tmp/microcosm-residual-uv-cache uv run
  --no-sync pytest -q`: `microcosm-frame`, `microcosm-fit`,
  `microcosm-calibrate`, `microcosm-data`, and `microcosm-build` all exited
  zero. The build shard reached 100% with one expected skip; only the
  established warning set appeared.
- The first build-shard pass exposed that the exact host command, when written
  literally in this journal, violated the live-tree retired-data-package guard
  (`packages/microcosm-build/tests/test_us_plan.py:966-1052`). The command now
  constructs those two host-only path segments from shell variables without
  changing their expansion. The focused guard and the entire build shard both
  pass on the corrected tree.
- Repository-wide `uv run --no-sync ruff check .` and `git diff --check` pass.
  No pool build, gate change, exclusion change, certification, push, or host
  publication was performed.
- After writing `FINAL_REPORT.md` and closing `PROGRESS.md`, reran all five
  package roots, repository-wide Ruff, the live-tree retired-data-package
  guard, and whitespace checks. Every command exited zero with only the
  established skips and warnings; the closeout commit changes documentation
  only relative to executable commit `eaba1eab`.

### Serial-host 1% rerun handoff

- `assembled` is the only semantically pre-change pkg3 cut: late calibration
  produces `transferred` (`multispine_pool.py:200-201`;
  `stacked_spine.py:10009-10037`;
  `tools/build_us_multispine_pool.py:3121-3175`). It cannot be mechanically
  warm-reused by the current CLI. The complete post-transfer policy is
  content-hashed (`post_transfer_calibration.py:297-344`) into stacked
  authority (`stacked_spine.py:2463-2492`), and configured identity routes
  checkpoints into a digest namespace
  (`tools/build_us_multispine_pool.py:1150-1178,4198-4227`). The parser exposes
  no audited stage-import override; the loader requires current manifest
  identity before choosing a valid stage
  (`tools/build_us_multispine_pool.py:406-532,1181-1250,1407-1428,1716-1937`).
- Therefore, do not copy, edit, rebind, or reuse pkg3's assembled/transferred/
  simulated manifests. A safe warm reuse would require a new audited
  stage-scoped import mechanism. The serial host owner should instead queue
  this exact cold 1% build in a fresh namespace under its existing single-
  build and `<15 GiB RSS` guard. Explicitly unsetting
  `POPULACE_LOGBOOK_PREV_ROW_DIGEST` prevents an ambient predecessor from
  entering the build (`tools/build_us_multispine_pool.py:3971-3979`). This lane
  did not run the command.

```bash
WT=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-residual-fixes
OUT=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/residual-fixes
DATA_REPO=/Users/maxghenis/PolicyEngine/policyengine-"us"-data
DATA_PACKAGE=policyengine_"us"_data

mkdir -p "$OUT"
cd "$WT" || exit 1

env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST \
  PATH="/Users/maxghenis/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  HOME=/Users/maxghenis \
  PYTHONUNBUFFERED=1 \
  "$WT/.venv/bin/python" tools/build_us_multispine_pool.py \
  --sample-fraction 0.01 \
  --sample-seed 578 \
  --clone-attachment-fraction 1.0 \
  --clone-attachment-seed 578 \
  --asec-raw-stage-h5 /Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5 \
  --asec-raw-stage-h5-sha256 51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe \
  --acs-household-zip /Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip \
  --acs-household-zip-sha256 8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0 \
  --acs-person-zip /Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip \
  --acs-person-zip-sha256 afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894 \
  --acs-rent-h5 "$DATA_REPO/$DATA_PACKAGE/storage/acs_2022.h5" \
  --acs-rent-h5-sha256 0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4 \
  --puf-h5 "$DATA_REPO/$DATA_PACKAGE/storage/puf_2024.h5" \
  --puf-h5-sha256 7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df \
  --puf-source-year-csv "$DATA_REPO/$DATA_PACKAGE/storage/puf_2015.csv" \
  --puf-source-year-csv-sha256 0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df \
  --checkpoint-root "$OUT/pool.checkpoints" \
  --out "$OUT/pool.h5" >>"$OUT/build.log" 2>&1

rc=$?
echo "residual-fixes exit: $rc" >>"$OUT/build.log"
exit "$rc"
```

## Main/F0 merge continuation — 2026-08-22 02:09Z

- Starting tracked revision: `c22e5d37`; local `origin/main`: `b4dfa0e7`.
- This continuation owns only the main merge, the F0 home for the existing
  post-draw calibration policy declaration, its spec/code identity test, and
  the resulting closed-world anti-rot updates. It does not run a host build,
  change a comparator/gate/threshold, certify an artifact, publish a release,
  or merge PR #742.
- Main's generated F0 bundle supersedes `specs/us_imputation_lineage.yaml`, so
  the old file will remain deleted. The policy itself must survive by becoming
  a closed, typed exact variant at the authored imputation-model location.
- Earlier notes below remain source-cited history. Their branch-state and
  handoff language is not current for this continuation.

## Scope and frozen boundary

This lane owns 16 adjudicated FIX-CANDIDATE checks: 13 in
`source_operator_two_part_calibration`, two in
`adult_care_post_reconciliation`, and one in
`model_required_targeted_calibration`. The assigned baselines and named
remedies are the rows in the read-only reference
`../microcosm-arm-split/experiments/battery_burndown/adjudication.json`.

No comparator, band, threshold, seed, fold, or sample contract may change. All
builds in this lane are off-chain at `--sample-fraction 0.01` and
`--sample-seed 578`; they omit `--logbook-prev-row-digest` and do not touch
`logbook-pending-chain.txt`.

## Post-transfer receipt failure #2: ordered capacity evidence

### No-build reproduction

- The SHA-pinned replay harness reads the assembled Frame checkpoint and the
  unemployment-compensation and weeks-unemployed target-bank H5 files,
  validates their file, identity, and raw-draw digests, reconstructs only the
  native clone-0 vectors, and calls the calibration kernel and its strict
  receipt validator. It performs no target fit, DAG execution, Frame write, or
  build (`tools/reproduce_us_post_transfer_weeks_checkpoint.py:1-6,74-112,115-250`).
- The stacked owner selects ASEC clone-0 reference rows, ACS clone-0 recipient
  rows, transferred-null mutable rows, and—for weeks—only mutable rows with
  positive unemployment compensation as allowed/addition candidates
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8960-8971,8995-9032`).
- The preserved replay contains 4,311 reference rows, 34,293 recipient/mutable
  rows, 134 positive reference rows, 24 initial recipient positives (all
  disallowed), and 32 positive-UC addition candidates. Its reference total is
  `80,851,529.27715749`, recipient total is `79,926,522.10879111`, reference
  positive mass is `2,762,659.3294707513`, and target positive mass is
  `2,731,052.2627107087` (`tools/reproduce_us_post_transfer_weeks_checkpoint.py:144-206,230-250`).
- At reproduction commit `4cc41652`, capacity generation reduces the 32
  candidate weights with masked `ndarray.sum`, producing
  `85,676.23791782455`, while selection independently reduces the same
  ID-ordered weights with `np.cumsum`, producing `85,676.23791782456`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:463-493,817-880` at that commit).
  The exact validator relationship `0 <= lower <= upper <=
  expected_candidate_mass` therefore rejects the receipt by one float64 ULP,
  `1.4551915228366852e-11`; every other relationship passes
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:1389-1490` at `4cc41652`).

### Semantic adjudication

- The invariant is correct and target-type agnostic: a selected prefix cannot
  exceed its declared candidate capacity. No tolerance or target exception is
  authorized. The generating defect is that capacity and prefix evidence are
  derived by two reduction schedules for the same declared ordered carrier
  set (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:321-327,463-493,817-891` at `4cc41652`).
- Weeks is a valid positive-carrier target. The ASEC source accepts only
  integer `-1` or `0..52`, maps `-1` to zero, and defines the positive event as
  an in-range integer above zero
  (`packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py:791-800,1218-1222`).
  Its dedicated QRF path rounds, clips, and revalidates predictions in that
  domain; post-transfer amount mapping then selects only positive
  reference-donor values and rejects support escape
  (`packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py:911-983`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:588-626,702-705`).
- The semantics-preserving repair is one immutable ordered-prefix schedule per
  candidate set, consumed by both capacity and selection. Rewriting only the
  terminal cumulative value is invalid because `_nearest_prefix` requires a
  nondecreasing vector for its lower-mass tie break and `searchsorted`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:487-515`).

### Two generating defects and the complete repair

- `_PrefixSchedule` binds one immutable ordered-position vector to one
  float64 cumulative-mass vector. `_nearest_prefix` consumes that schedule and
  reports its terminal element as candidate mass
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:282-287,471-515`).
  Commit `d7b12bab` constructs the amount-descending/ID removal schedule and
  ID-ordered addition schedule once, so capacity and selection use the same
  candidate endpoint (`post_transfer_calibration.py:844-872,891-928`).
- The SHA-pinned weeks replay therefore changes its candidate mass from
  `85,676.23791782455` to the selection schedule's exact terminal
  `85,676.23791782456`; upper minus candidate becomes zero and strict
  validation succeeds. The harness now pins both the failing and repaired
  relationship values rather than accepting an arbitrary validator failure
  (`tools/reproduce_us_post_transfer_weeks_checkpoint.py:210-286`).
- The required cross-target replay found that `d7b12bab` still composed its
  whole maximum from independently rounded partition endpoints. Both actual
  child-support targets produced maximum `79,926,522.10879174` against
  recipient total `79,926,522.10879111`, so only the exact
  `maximum_attainable_mass <= recipient_total` relationship failed, by
  `6.258487701416016e-07`. Expense partitions were
  `71,696.09739141785 + 79,854,826.01140033`; received partitions were
  `180,209.75664861224 + 79,746,312.35214312`.
- The complete generating repair declares the attainable carrier set once as
  `fixed_positive | allowed_positive | zero_candidates`. It zero-masks that
  set onto the already ordered recipient-weight vector, retaining the same
  vector length and reduction topology used by `recipient_total`. For
  nonnegative weights, the exact subset bound is therefore structural; the
  maximum is neither a sum of rounded partition scalars nor a clamp
  (`post_transfer_calibration.py:823-885`).
- Both SHA-pinned child receipts now have maximum exactly equal to recipient
  total `79,926,522.10879111` while the historical partition sum remains
  `79,926,522.10879174`; every strict relationship passes. The child harness
  pins both targets' file/identity/raw hashes and requires the exact per-target
  state, error, relationship, and floats on the red and green sides
  (`tools/audit_us_post_transfer_child_support_checkpoints.py:1-80,91-207,210-302`).
- A proper-subset regression supplies weights for which compressed regrouping
  yields `0x1.433526fbe1946p+48`, `0.0625` above recipient total
  `0x1.433526fbe1945p+48`. The same-topology union yields the recipient value
  exactly and validates for every late `match_reference` declaration. Separate
  regressions cover the production weeks candidate bytes, independently
  rounded whole partitions, and the symmetric removal path
  (`packages/microcosm-build/tests/test_us_post_transfer_calibration.py:544-753`).
- The validator is unchanged: it still requires exact
  `maximum_attainable_mass <= recipient_total` and exact
  `0 <= lower <= upper <= candidate_mass`; its pre-existing approximate
  partition-additivity and boundary checks were not adjusted
  (`post_transfer_calibration.py:1457-1529`). No tolerance, threshold, band,
  gate, or target exception changed.

### Complete late-transfer target audit

The registry contains exactly seven late targets; six share the repaired
`match_reference` branch and one bypasses carrier selection by preserving
recipient carriers
(`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:208-258,786-812,840-932`).

| Late target | Source semantics / observed 1% shape | Carrier verdict |
| --- | --- | --- |
| `pre_subsidy_care_expenses` | Nonnegative monetary care expense. ACS reconciliation restricts it to qualifying people and at most one carrier per tax unit; the late owner admits existing qualifying positives and one stable zero candidate per empty unit (`acs_transfer.py:660-739,1277-1299`; `stacked_spine.py:8728-8746,8977-8986`). | Covered: constrained `match_reference`; not a count. Its proper attainable subset uses the same whole-union mechanism and proper-subset regression. Host had not reached it. |
| `child_support_expense` | Exact nonnegative annual `CHSP_VAL` carry (`child_support.py:166-201`); QRF reported six distinct donor values. | Covered: monetary `match_reference`. Its pinned checkpoint fails at `d7b12bab` and passes the complete union repair. |
| `child_support_received` | Exact nonnegative annual `CSP_VAL` carry (`child_support.py:166-201`); QRF reported 15 distinct donor values. | Covered: monetary `match_reference`. Its pinned checkpoint fails at `d7b12bab` and passes the complete union repair. |
| `disability_benefits` | Nonnegative annual two-slot sum excluding workers' compensation (`disability_benefits.py:184-220,558-560`); QRF reported ten distinct donor values. | Inapplicable: `preserve_recipient` emits neither capacity nor selection evidence (`post_transfer_calibration.py:230-236,1319-1335`). Its preserved checkpoint validates with before/after carrier mass `42,658.57948297383`. |
| `weeks_unemployed` | Integer `-1` or `0..52`, with `-1` mapped to zero (`weeks_unemployed.py:791-800,911-983,1218-1222`); QRF reported 12 distinct donor values. | Covered: sole semantic count; positive-UC-constrained `match_reference` (`stacked_spine.py:8995-9008`). The exact replay proves reducer order, not count support, caused the failure. |
| `workers_compensation` | Exact nonnegative annual `WC_VAL` carry (`workers_compensation.py:143-184,520-522`); host had not reached it. | Covered: monetary default-mask `match_reference`; shared-kernel regressions validate its declaration. |
| `spm_unit_energy_subsidy` | Measured nonnegative annual `SPM_ENGVAL`, checked within unit and reduced to SPM-unit float64 (`energy_subsidy.py:157-233,543-557`); host had not reached it. | Covered: monetary default-mask `match_reference`; shared-kernel regressions validate its declaration at its entity grain. |

The host log's near-discrete evidence appears at `build.log:1252-1266,1404-1408`.
Weeks does not rely on ACS's explicit discrete-numeric set, which contains only
two mortgage-year targets; ordinary numeric targets otherwise use the
continuous encoding. Its integer semantics come from the dedicated weeks
source/QRF checks and post-transfer donor-support mapping cited above
(`acs_transfer.py:129-138,3035-3117`). QRF's `<=32`-unique “near-discrete”
branch is a leaf-storage optimization, not a carrier-capacity semantic
distinction, which is why annual dollar targets also triggered it here
(`microcosm-fit/qrf.py:388-401,482-503`).

Current zero-based late-DAG positions are child support 24, disability 25,
weeks 30, workers' compensation 31, energy subsidy 32, and adult care 34.
The registry constructs and schedules those groups deterministically, stacked
execution enumerates them serially, and each production group applies its
post-transfer calibration before returning
(`us_late_producer_registry.py:1338-1396,2013-2019`;
`stacked_spine.py:10054-10095,10927-10931`). The failed host run had crossed
child support and disability but had not produced checkpoints for workers'
compensation, energy subsidy, or adult care. Their verdict is therefore a
source/mask proof plus the six-spec shared-kernel regressions, not a claim of
checkpoint replay.

## Source-cited mechanism record

- The battery computes positive and negative carrier incidence separately,
  compares ACS/ASEC weighted incidence to the frozen band, then computes five
  weighted conditional carrier quantiles when both sides have enough rows
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:13351-13421`).
- Its quantile-envelope diagnostic is the maximum symmetric normalized
  separation across those five conditional quantiles
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:13481-13512`).
- The fitted QRF draws a weighted sign gate and then a forest amount conditional
  on the selected sign; that is the two-part mechanism calibrated here
  (`packages/microcosm-fit/src/microcosm/fit/qrf.py:950-1003,1333-1429`).
- Ordinary ACS transfers partition recipients by optional-predictor
  availability and construct exact complete-donor model frames. Regime
  detection and fitted-model verification run only for an explicit
  owner-selected target subset; the existing pattern seed and draw surface do
  not change
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:919-1115,1297-1579`).
- The banked path applies the same explicit selection before the targetwise
  chain and verifies only selected returned target regimes before accepting
  their raw draws
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1621-1848`).
- Adult-care qualification is a fail-closed section-21 predicate, and the
  reconciliation clears nonqualifying mutable carriers, permits at most one
  qualifying mutable carrier per tax unit, and preserves pre-existing positive
  carriers
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:660-732`).
- The weeks-unemployed signal gate rejects positive PUF support without
  unemployment compensation, so carrier additions must respect that compatible
  capacity (`packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py:1328-1358`).
- The prior-year income gate explicitly requires the ASEC negative
  self-employment leg to survive, so this lane may calibrate only the positive
  mutable leg (`packages/microcosm-build/src/microcosm/build/us_runtime/prior_year_income.py:829-887`).

These mechanisms support an artifact-side correction; they do not justify a
comparator change.

## Implemented artifact mechanism

- The immutable nine-target policy declares the exact early/late owner, carrier
  mode, byte-exact negative leg, adult-care constraint, and weeks/UC constraint
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:123-263`).
- The kernel requires disjoint reference/recipient masks and mutable-subset
  scope, snapshots and byte-compares protected surfaces, computes the reference
  positive mass, and uses deterministic nearest-prefix removal/addition within
  a proven attainable-mass interval
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:445-547,727-922`).
- Mutable positive amounts are mapped only to reference positive support and
  are anchored at the frozen 10/25/50/75/90 percentiles; infeasible or
  conflicting anchors are recorded rather than hidden
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:577-724`).
- The kernel proves boundary saturation when capacity-limited and rejects any
  change to nonmutable, negative, negative-zero, or zero-weight bytes, any
  donor-support escape, or any preserve-carrier change
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:939-978`).
- The stacked owner derives ASEC clone-0 reference rows, ACS clone-0 recipient
  rows, and transferred-null mutable rows. Adult care uses qualifying rows plus
  one candidate per empty unit; weeks uses positive-UC mutable rows. The final
  adult-care reconciliation must be byte-identical/no-op
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8799-8955`).
- Schema-v2 receipts explicitly state that terminal validation cannot replay
  pre-calibration state; they separate live-replayable output claims from
  generation-transition evidence
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:43-119,980-1055`).
- Terminal validation independently recomputes live row/mask hashes, entity and
  output hashes, full weights, carrier masses, reference/recipient quantiles,
  QED, and coupled adult/weeks constraints
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3756-3818,3821-4025`).
- Only the nine declared calibration targets receive QRF pattern receipts.
  Those receipts persist ordered predictors, seeds, weights, row counts, and
  selected regimes, then validate canonical predictor/pattern/target order and
  exact record-family binding. All assigned and unassigned receipts validate
  the same legacy row-count schema and accounting before their scope branch.
  Unassigned transfers retain their evidence-free receipts and generic
  serializers omit the empty opt-in field. Receipt-only validation deliberately
  does not claim donor replay or out-of-sample verification
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4128-4445,4529-4795`).
- Canonical early and late production both use the certified maximum of eight
  targets per fit. Exact selected-family receipt binding therefore cannot be
  weakened by caller-selected batching; the non-production test seam retains
  smaller-width coverage.
- The two pinned 3.73 GB SIPP readers retain chunked selection and downstream
  explicit coercion while using streaming type inference. Guarded full-donor
  reruns observed much lower RSS and unchanged locked donor facts
  (`packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:393-413`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/voluntary_filing.py:375-397`).

## Adjudication limits retained

- The named adjudication remedy calls for target/sign-scoped origin-aware
  cross-fitted carrier calibration and held-out nonregression. This branch does
  not have the adjudicated fold/comparator authority in main and does not invent
  one. Its carrier correction is deterministic terminal reference-margin
  matching, not cross-fitting or an out-of-sample estimate
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:839-922`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4211-4217`).
- The unemployment-compensation row likewise lacks the adjudicated money-OOS
  authority in this branch. The implementation freezes its carrier membership
  and calibrates only conditional positive amounts; no OOS nonregression claim
  is made
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:193-207,952-978`).
- Weeks-unemployed carrier matching is allowed to stop at the exact
  positive-UC-compatible capacity, but the receipt must prove the attainable
  interval and boundary saturation
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:839-955`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8863-8876`).
- Mutable masks, input hashes, before-state diagnostics, change counts, and
  byte-preservation proofs need the generation-time pre-frame. Terminal
  validation authenticates them through the enclosing execution authority and
  does not claim to reconstruct them from the final frame
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:43-119`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3925-3938`).

## Environment receipt

- `uv sync --all-packages --extra us`: failed before resolution because the
  sandbox denied writes to `/Users/maxghenis/.cache/uv`.
- Writable-cache retry: failed downloading `pandas==3.0.3` because sandbox DNS
  is unavailable.
- Exact-lock fallback: `uv.lock` SHA-1
  `6b213e740b114d008c0191fa492832a957a0a948` matches
  `../microcosm-707/uv.lock`; that environment imports NumPy 2.4.6, pandas
  3.0.3, and pytest 8.4.2 while `PYTHONPATH` points at this worktree.
- The pre-continuation PR test surface was green: all 225 `microcosm-build`
  test files; `microcosm-fit` 93 passed; `microcosm-calibrate` 201 passed;
  `microcosm-frame` 294 passed/36 skipped; and `microcosm-data` 275 passed/one
  skipped. Heavy files ran in fresh pytest processes. The final continuation
  tree also passes all 225 `microcosm-build` test files across fresh processes,
  including full ordinary/banked transfer, stacked binding, serializer, pool,
  H5 loader, and real late-executor files. The full rerun exposed one synthetic
  H5 target receipt with only `residual_null_rows`; its fixture was updated to
  the valid four-zero count block, its full file reran green, and a fixture
  scan found no other canonical partial blocks.
- `ruff check .`, touched-file `ruff format --check`, and `git diff --check`
  pass. Full-tree format checking reports 49 pre-existing files outside this
  lane's formatting scope.
- On the completed post-transfer receipt repair, all 47 focused calibration
  tests and all five package roots (`microcosm-fit`, `microcosm-calibrate`,
  `microcosm-data`, `microcosm-frame`, and the complete `microcosm-build`
  root) exited zero under the guard. Repository-wide Ruff, touched-file
  formatting, and whitespace checks also pass. Only established skips and
  warnings appeared; no host build ran.
- The final successful full-donor tests peaked at 0.485 GiB for vehicles and
  0.532 GiB for voluntary filing; the largest successful isolated build-test
  shard peaked at 6.531 GiB. An earlier 13.5 GiB/250 ms diagnostic guard
  observed one rapid parser spike at 15.424 GiB before termination. The guard
  was immediately tightened to 10 GiB/20 ms, the streaming reader was fixed,
  and all successful reruns stayed below the cap.

## Measurement ledger

The before build used `--sample-fraction 0.01 --sample-seed 578`, clone fraction
1 and clone seed 578, with no chain predecessor. Peak observed per-process RSS
was 10.733 GiB. Artifact receipts:

- `pool.h5`: SHA-256 `258891504201275f8006a1584b7d3e891890d15381724bc9f2b30b1f443d967f`
- `pool.gates.json`: SHA-256 `94ee914bb7490e7f513184e691cf15847d1e585693ef184977196485f86f1fee`
- `pool.manifest.json`: SHA-256 `953aaff72fac8cc17211959d0133f9bce8c22dd87b2957a0f6a89335fcc9c122`

The exact frozen-battery values are below. `unsupported` means the battery did
not emit QED because one side had fewer than five carriers; the parenthesized
number is the same five-grid weighted diagnostic computed manually without
changing that support rule.

| Assigned check | Before at 1% |
| --- | ---: |
| adult care positive incidence ratio | 0.561425035 |
| adult care positive QED | unsupported (manual 1.738865343) |
| unemployment compensation positive QED | 0.352941176 |
| child-support expense positive incidence ratio | 0.171280844 |
| child-support expense positive QED | 0.953846154 |
| child support received positive incidence ratio | 0.242882414 |
| child support received positive QED | 1.000000000 |
| disability benefits positive QED | 1.373534621 |
| prior-year self-employment positive incidence ratio | 1.376752468 |
| prior-year self-employment positive QED | 0.834720589 |
| weeks unemployed positive incidence ratio | 0.025384419 |
| weeks unemployed positive QED | 0.736842105 |
| workers' compensation positive incidence ratio | 0.047614655 |
| workers' compensation positive QED | unsupported (manual 1.918367347) |
| SPM-unit energy subsidy positive incidence ratio | 0.240477284 |
| SPM-unit energy subsidy positive QED | 0.666666667 |

Sparse 1% sampling also puts the otherwise frozen unemployment-compensation
and disability positive carrier ratios at 0.131859569 and 0.098471480. Their
adjudicated remedies freeze carrier membership, so this lane will improve only
their assigned amount-QED checks. The unassigned negative prior-year
self-employment ratio is 1.435953092 and is likewise deliberately untouched.

After values will be added after the calibrated 1% rebuild.
