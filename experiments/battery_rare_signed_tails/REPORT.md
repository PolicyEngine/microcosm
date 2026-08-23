# Rare signed-tail battery lane — adjudicated report

Branch `battery-rare-signed-tails`, cut at `2c7a7218` (`origin/main`).
Authority: the frozen arm-split adjudication
(`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/experiments/battery_burndown/ADJUDICATION.md`
and `adjudication.json`) — 48 red QED checks plus the one-sided Keogh leg.
Every line-number citation below is against this branch's tree unless marked
otherwise. No gate, incidence band, QED ceiling, support floor, training-cap
value, fold, seed value, exclusion register, or logbook chain changed on this
branch. The lane started no pool build; the host queue owns all builds.

## 1. Classification of the 48 red checks

[`realized_regimes.json`](realized_regimes.json) classifies every check on
the three mechanism dimensions the charter requires, recomputed from the
frozen fit boundary per the adjudication's remediation rule (recompute the
regime from frozen donor support for each availability pattern before any
regime-specific fix; ADJUDICATION.md §Comparator mechanisms, receipt-gap
bullet). The extractor
([`build_realized_regime_evidence.py`](build_realized_regime_evidence.py))
refuses changed authority hashes, changed pattern inventories, donor/support
non-closure, or a changed 48-check/42-target/168-record counting identity.

**Fitter regime (recomputed, not assumed).** At `zero_atol = 1e-6` every one
of the 48 checks realized a *gated* regime at its fit boundary: 35
`zero_inflated_positive` and 13 `three_sign` (42 unique targets: 35/7). None
is `degenerate_zero`, single-sign, or sign-only. All 42 target checkpoints'
banked draws are sign-consistent with these regimes, and all seven
`three_sign` targets show both signed classes in their decoded draws. A
mis-realized regime therefore explains **zero** of the 48 failures, and no
sign-conditional draw change is honest for any of them — the sign machinery
engaged exactly as designed
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:92-150,950-1003,1333-1380`,
main-numbering citations from the adjudication).

**Transfer partition.** 17 checks are early ASEC→ACS gap-fill targets and 31
are late producer-complement targets (20 PUF-source, 10 ASEC-source, one
overlapping producer target). The partition is ownership context for the
remedy, not itself a defect.

**Donor-support starvation.** Eight checks are support-starved; forty are
not. The starved eight split into two causally distinct classes: five whose
route donors are sparse but *intact* (blocked on additional evidence per the
adjudication), and three whose support was *deleted upstream* by the
retirement training cap (fixed on this branch, §2.1).

Full per-check listing, by mechanism class:

| Class (mechanism.primary) | Checks | Ordinals | Status |
|---|---:|---|---|
| gated_conditional_magnitude_shape | 23 | 9, 10, 12, 13, 14, 18, 30, 32, 35, 36, 38, 40, 42, 43, 44, 48, 49, 50, 52, 55, 56, 90, 92 | blocked_on_owner_specific_held_out_evidence |
| source_operator_conditional_magnitude_shape | 7 | 68, 70, 71, 73, 84, 86, 88 | blocked_on_owner_specific_held_out_evidence |
| donor_evidence_starvation | 5 | 16, 28, 33, 46, 75 | blocked_on_additional_evidence |
| retirement_mapping_label_or_model_shape | 4 | 21, 23, 25, 26 | blocked_on_mapping_labels_and_dense_evidence |
| upstream_retirement_support_deletion | 3 | 78, 80, 82 | **implemented** |
| late_transfer_then_qbi_reconciliation_shape | 2 | 58, 60 | blocked_on_qbi_owner_and_held_out_evidence |
| late_transfer_conditional_magnitude_shape | 2 | 62, 64 | blocked_on_held_out_calibration_evidence |
| adult_care_post_transfer_amount_shape | 1 | 2 | blocked_on_held_out_calibration_evidence |
| early_transfer_unemployment_amount_shape | 1 | 8 | blocked_on_held_out_calibration_evidence |

## 2. Smallest honest change per class

The class-level answer is explicit below. Only the support-deletion repair is
authorized by the frozen evidence; the other eight changes are the smallest
code seams that would fix their generating mechanisms *after* the named owner
evidence exists. Applying them now would be target tuning toward the battery.

| Class | Smallest generating-mechanism change | Disposition |
|---|---|---|
| gated_conditional_magnitude_shape | Apply a target-and-sign weighted magnitude map only to imputed recipient draws at the owning QRF boundary (`acs_transfer.py:1348-1413`). | Owner-held-out evidence required. |
| source_operator_conditional_magnitude_shape | Apply the same target/sign-scoped amount map inside the owning registered source-family callback, preserving producer-owned cells and its reconciliation (`stacked_spine.py:5030-5278`). | Source-family owner evidence required. |
| donor_evidence_starvation | After a source-label audit provides dense evidence, widen the authorized donor rung used by the pattern donor mask or fit a target-specific rare-event/sign-and-magnitude model; do not lower the support floor (`acs_transfer.py:1351-1385`; `stacked_spine.py:3027-3031`). | Additional evidence required. |
| retirement_mapping_label_or_model_shape | Correct an evidenced account/source-label mapping, then refit the applicable amount model; the current account-code map and SS predictor seam are explicit (`retirement_distributions.py:190-208,679-691`). | Mapping audit plus dense rung required. |
| upstream_retirement_support_deletion | Retain the union of four-target carriers before the fixed cap and reweight only sampled common zeros (`retirement_distributions.py:443-450,486-584`). | **Implemented.** |
| late_transfer_then_qbi_reconciliation_shape | Calibrate the joint pre-cap QBI surface while preserving the existing coupled identities and caps (`qbi_inputs.py:1228-1363`). | QBI owner and held-out evidence required. |
| late_transfer_conditional_magnitude_shape | Apply a target-scoped weighted magnitude map to mutable late-transfer draws only (`acs_transfer.py:1398-1413`). | Held-out evidence required. |
| adult_care_post_transfer_amount_shape | Calibrate positive amounts on newly imputed mutable cells before the existing adult-care reconciliation (`acs_transfer.py:1244-1265`). | Held-out evidence required. |
| early_transfer_unemployment_amount_shape | With predictors and carrier membership frozen, apply a target-only weighted positive-magnitude map in the early QRF draw path (`acs_transfer.py:1348-1413`). | Held-out evidence required. |

### 2.1 Upstream retirement support deletion — implemented

Ordinals 78/80/82 (taxable 401k/403b/SEP positive) and the Keogh leg share
one generating mechanism. The retirement producer trains a four-target QRF
on the 108,073-row native ASEC donor and replaces the four non-IRA outputs
on the PUF half of the `asec_puf` donor spine
(`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:337-483`).
The old cap drew a *uniform* 5,000-row sample, which retained only
102/2,057 native 401k carriers, 4/161 403b, 2/61 SEP, and 0/2 Keogh
(`realized_regimes.json` → `retirement_cap_support`). The late ACS transfer
then draws its donors from that producer-overwritten PUF channel — the
`asec_puf` spine with the AUTO channel resolving to PUF support
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:118-123`;
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8582`)
— so deleted carriers can never reach ACS.

The fix repairs the sampling design inside the unchanged cap:
`_cap_retirement_distribution_training`
(`retirement_distributions.py:486-586`, called at `443-450`) retains every
row nonzero for *any* of the four targets as a census, samples only the
common all-target-zero stratum under the same 5,000-row cap and the same
seed, prefers positive-weight zero rows, ratio-calibrates only the sampled
zero rows back to the full zero-stratum mass (carrier weights and
conditional magnitudes stay exact), and fails closed on nonfinite/negative
inputs or a carrier union that leaves no zero slot. `sources.yaml:2138`
still declares `max_train_samples: 5000`; no cap value, seed, or gate
moved. The seed-protocol declaration records the changed draw-consumption
contract
(`packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:877-901`).

### 2.2 Donor evidence starvation — blocked, correctly

Ordinals 16 (collectibles +), 28 (alimony income +), 33 (casualty loss +),
46 (farm operations −), 75 (prior-year self-employment −). Recomputed
route-donor sign support: collectibles +18 (late PUF; +82 native ASEC),
alimony +61 (early), casualty +27 (late PUF; +79 native), farm operations
−89 (early), prior-year self-employment −48 (early). Every donor is gated
and sign-capable, all above the five-carrier floor
(`stacked_spine.py:3027-3031`), so no transfer defect deletes them — the
tails are simply too sparse for the fitted conditional magnitudes to match.
The adjudication requires dense/additional evidence before a
target-specific refit; a code change here would be tuning toward the gate.
These remain BLOCKER, owner-scoped (dense-rung donor evidence or an
owner-approved target-specific sparse-tail model).

### 2.3 Shape classes — regime receipts closed; calibration blocked

The remaining 40 checks (classes `gated_conditional_magnitude_shape`,
`source_operator_conditional_magnitude_shape`,
`retirement_mapping_label_or_model_shape`,
`late_transfer_then_qbi_reconciliation_shape`,
`late_transfer_conditional_magnitude_shape`,
`adult_care_post_transfer_amount_shape`,
`early_transfer_unemployment_amount_shape`) fail on conditional magnitude or
incidence *shape* with intact gated donors. The adjudication's only
executable defect here was the receipt gap: ACS-transfer provenance did not
persist the realized regime, so signed-tail fits could not be adjudicated
without reconstruction. This branch closes that gap end to end:

- every ordinary fit binds validated realized regimes into provenance
  (`acs_transfer.py:52-85,453,1392-1396`), and every banked chain step
  persists its regime (`acs_transfer.py:498,1819-1823`);
- target-bank schema 2 stores and revalidates the regime per pattern step
  and projects it into operational receipts
  (`acs_transfer_bank.py:25,47-53,249,357`); schema-1 artifacts rebuild;
- early-direction, late-group, and aggregate receipts expose the same
  fitted-pattern regime rows, and the canonical aggregate must be the exact
  concatenation of the 19 group receipts
  (`stacked_spine.py:3754-3772,8215-8360,9260-9268,9315-9331`).

The magnitude repair itself (recalibrating conditional draws against
held-out evidence, or fixing SS/retirement mapping labels) is owner-scoped
calibration work that the frozen evidence does not authorize; each row's
`mechanism.smallest_honest_change` records the specific blocked remedy.

## 3. The Keogh one-sided leg: lost before transfer fitting, not structurally absent

Terminal leg
`person/source_operator_retirement_distributions/keogh_distributions[clone_0]/positive`;
full-scale f025 emits `weighted positive-leg incidence ratio 0 is outside
[0.8, 1.25] (asec=1.43573e-05, acs=0)`. Incidence runs and fails *before*
the five-carrier floor, which skips only the sign's QED
(`stacked_spine.py:12068-12087`), exactly as the adjudication states.

Determination: **the ACS carriers are lost on the transfer donor path — fix,
not declare-absent.** More precisely, the retirement producer deletes them
before the ACS-transfer fit rather than the recipient merge dropping them.
The evidence chain, each step verified against frozen artifacts this lane:

1. Native ASEC clone 0 holds exactly two positive Keogh values, `2,040` and
   `30,000`, in 108,073 rows — the signal exists at the source.
2. The old uniform 5,000-row cap retained neither carrier
   (`retirement_cap_support.keogh_distributions: native 2, retained 0`), so
   the producer's QRF predicted identically zero Keogh across the entire
   PUF half.
3. The frozen transfer bank holds 1,970,973 stored / 1,736,840 finite Keogh
   recipient draws, every one exactly zero (`max_abs = 0.0`) — the fitted
   transfer regime was `degenerate_zero` *because its donor channel was
   already dead*, not because the recipient merge dropped values.
4. Transferred-frame census: ASEC clone 0 keeps `{2,040, 30,000}`; ASEC
   clone 1 and every ACS clone are all-zero. The ACS absence is
   manufactured upstream of transfer fitting.

The declared-absence route would have been honest only if ACS Keogh were
structurally impossible. It is not, so that route is rejected. For the
record, taking it would require: an exact structural-absence equation for
the recipient scope (validated as exact, with zero unexpected-null and zero
synthesized rows, `stacked_spine.py:8168-8195`) or a
`tolerated_absence_receipts` declaration on the late-producer requirement
with its canonical per-requirement receipt
(`stacked_spine.py:6491-6528`), plus battery-side structural receipts that
stay clone-0 exact (`stacked_spine.py:7186-7205`). Declaring absence for a
present-in-source signal would also delete the pool's only terminal check
of it — the same reasoning the adjudication used to reject a clone-0 waiver
(ADJUDICATION.md line 71). Approval would sit with the US pool owner via
the owner-only reviewed-exclusion/register path, which this lane is barred
from and did not touch.

With the support-preserving cap, both carriers now reach the producer's
training donor whenever the full donor is available;
`test_production_imputer_hands_rare_keogh_support_to_qrf` locks that
production call path.

## 4. Verification

### 4.1 Regression tests (all on this branch)

`packages/microcosm-build/tests/`:

- `test_us_retirement_distributions.py` —
  `test_production_imputer_hands_rare_keogh_support_to_qrf`,
  `test_puf_training_cap_preserves_nonzero_union_and_weighted_sign_mass`,
  `test_puf_training_cap_fails_when_nonzero_union_leaves_no_zero_slot`,
  `test_puf_training_cap_does_not_waste_representative_slots_on_zero_weights`.
- `test_us_acs_transfer.py` — three-sign regime recording, exact-target /
  known-label receipt closure, cold/warm bank regime persistence, schema-1
  rebuild.
- `test_us_stacked_spine.py` / `test_us_late_producer_dag.py` — late
  executor rejects a regime receipt detached from fitted provenance;
  aggregate is the exact 19-group concatenation; canonical JSON key sorting
  preserves valid regime-map checkpoints while a missing target still fails
  (`test_realized_regime_receipt_survives_canonical_json_key_sorting`).
- `test_us_multispine_pool_h5_io.py` / `test_us_multispine_pool_tool.py` —
  legacy materializer-3 checkpoints cannot resume after the shared fix;
  stacked v12 and pool v8 identity binding; v7 rejection.
- `test_spec_engine_seeds.py` — the support-preserving cap's draw-site
  declaration.

### 4.2 Suite receipts

Per-shard `uv run pytest` (one process per shard, as CI runs it) on the final
source tree:

- calibrate: 201 passed;
- data: 275 passed, 1 skipped;
- fit: 93 passed;
- frame: 294 passed, 36 skipped;
- build: 5,973 passed, 39 skipped;
- repo-wide `uv run ruff check .`: all checks passed.

The authoritative US-bundle generator check and the frozen realized-regime
evidence generator check also pass. The build shard was rerun in full after
correcting its sole stale derived-payload assertion, so the receipt is not a
composition of partial runs. Exact durations and warning counts are in
PROGRESS.md. `ruff format` was applied only to the five files whose drift this
branch introduced; the CI lint gate is `ruff check` and is green.

### 4.3 1% before/after

The host-queue-owned baseline 1% build
(`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/baseline1pct/pool.gates.json`)
is the frozen "before":
[`baseline_1pct_projection.json`](baseline_1pct_projection.json), generated
by [`project_baseline_1pct.py`](project_baseline_1pct.py), shows 127
battery failure lines over 93 legs; 47 of the 48 frozen reds reproduce at
1%; ordinal 16 (collectibles) emits no 1% line because its +18 route
carriers rarely survive a 1% sample; the Keogh leg emits no 1% line in
either direction because a 1% ASEC sample almost surely carries neither
native carrier.

Regenerating the projection on this final tree was byte-identical to the
committed JSON. Running [`diff_1pct_failures.py`](diff_1pct_failures.py) with
that baseline on both sides reports 127 shared lines and zero added/removed or
changed lines, which validates the diff path without inventing an after result.

Consequences, stated before the after-build exists: at 1% the fix is
falsifiable on the 401k/403b/SEP legs (native carriers ~20/1.6/0.6 expected
in-sample) and on non-regression of the other 89 legs; the Keogh gate-level
flip is only provable at full scale, where its regression test and the
bank-draw census stand in until the owner's next full build. This lane is
barred from starting builds, so the "after" 1% run is a host-queue action:
[`run_1pct_offchain_build.sh`](run_1pct_offchain_build.sh) (1% cap,
off-chain, serial-queue refusal, 15 GiB RSS watchdog) followed by
[`diff_1pct_failures.py`](diff_1pct_failures.py) `--before-gates
…/baseline1pct/pool.gates.json --after-gates <new build>` produces the
failure-line diff projected against the frozen baseline classes.

## 5. Identity and receipt version moves

All moves are semantic invalidations of artifacts that could otherwise
resume with support-deleting samples or regime-free receipts — none touches
a gate:

- late-producer receipt schema 3→4
  (`us_late_producer_registry.py:129`; `us/spec/imputation.yaml`
  execution_receipt_contract);
- legacy pool checkpoint materializer 3→4
  (`tools/build_us_multispine_pool.py:231`, mirrored at `h5_io.py:103`);
- pool stage checkpoint materializer 7→8
  (`tools/build_us_multispine_pool.py:282`);
- stacked checkpoint materializer 11→12
  (`tools/build_us_multispine_pool.py:319`; `us/spec/spine.yaml`;
  `spec_engine/schema/spine.schema.json`);
- ACS-transfer target-bank schema 1→2 (`acs_transfer_bank.py:25`).

Because `seeds.py` is shared spec-engine kernel source, its attested
identities legitimately moved and were re-pinned from the final tree: US
bundle `5b0014c3eb6cb121f0a9f2138ab860be30ef2251dd6ff4ec38cbf5f778899554`,
BE `91d3a58e669950c7fab0aefeb74687278efca82d413a900e9ab1878b09a7a5a3`,
UK `54c0b70167b9d36e857a116ca4c80c74af142b6f09d5894ee979de62b9634738`,
loader golden vector `43989f610f9691daf69e9ee74c1aa17f2a8bc3f07b9992489c65fa8b1ad747de`,
inventory `seed_protocol`
`d317a1cb0dc24ee49e528ac3a0d3337952b30c3e6610fd49026770a42b88d6be` and
`seed_map`
`f9e71d22a4c8608d1e2ca99912f60dd06272e785e0babcca46a452c1a6df9562`;
`docs/evidence/spec-engine/us-f0-coverage.json` regenerated by
`tools/spec_engine_coverage.py` (41,381/41,381 fields, 40/40 inventory
checks). An earlier journal claim of US bundle `d30642501b9b…` described an
intermediate tree and is historicized in `_LANE-NOTES.md`.

## 6. What remains (owner actions)

1. **Host queue:** run the 1% after-build from this branch
   (`run_1pct_offchain_build.sh`) and file the `diff_1pct_failures.py`
   output; expected movement is on 401k/403b/SEP legs plus no new
   non-baseline reds beyond 1% sampling noise.
2. **Full-scale build (owner-scheduled):** the only gate-level proof of the
   Keogh flip and the final word on ordinals 78/80/82.
3. **Owner calibration program:** the 35 blocked shape checks need
   held-out/dense evidence per class (§2.3); the five starvation blockers
   (§2.2) need dense-rung donor evidence or an owner-approved sparse-tail
   model; the four SS-mapping checks additionally need label adjudication.
4. **No exclusions were added** and none are recommended; the Keogh
   declared-absence route is explicitly rejected (§3).
