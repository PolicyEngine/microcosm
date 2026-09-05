# UK dataset sizes: implementation plan and operating boundary

This implements the candidate-building part of [#355](https://github.com/PolicyEngine/microcosm/issues/355)
on [#870](https://github.com/PolicyEngine/microcosm/pull/870)'s branch. The PR is still open as of
2026-09-05 and itself stacks on #852. Do not base the work on the old issue's
535,080-household 2023 dataset. The authoritative inputs are the current raw-FRS
2024-25 spine, OA ladder, and pinned Chronicle facts used by the joint candidate.

## Decisions retained

- Pool generation keeps clone count **K=15**. Requested output households is a
  different parameter, applied after cloning and materialization.
- The joint surface retains national, constituency and local-authority rows,
  the declared stretch bound **10**, loss cap **10**, **grain_equal** weighting,
  and **1,500 epochs** per solve in the normal driver defaults.
- Existing binding adjudications, signed deferrals, measure exclusions,
  census-vintage uprating, and all local gates remain in force. Size selection
  does not silently drop target rows, loosen ESS floors or change registers.
- Population-normalized engine measures are frozen from the full pool. The
  refit uses their selected household contributions rather than re-running
  those formulas on a smaller population.
- The Frame carries complete households, benefit units and people; every
  non-dry attempt retains the existing Logbook recording envelope.

## Implemented sequence

1. Build the usual joint dense solve from the pinned inputs. This also supplies
   the same-target reference loss for the size comparison.
2. Compute each household's maximum absolute target-contribution share from
   original pool weights and the compiled sparse matrix, following
   [#346's correction](https://github.com/PolicyEngine/microcosm/issues/346#issuecomment-4902880142).
   Protect the largest absolute weighted carrier of each nonzero target, with
   first-column tie breaking. Initialize open probabilities with
   `0.1 + 0.8 * score / (score + median_positive_score)`. This bounded smooth
   prior is an implementation choice for candidate evaluation, not a measured
   UK release ruling. Run L0 budget search; never select the largest prior scores.
3. Use the existing exact-count Sampford sampler on learned probabilities.
   Probability-one gates are certainties (`pi_hi=1`), including every protected
   carrier. Refuse impossible budgets or sampling designs rather than clamp.
4. Refit on the selected support through `microcosm.calibrate`, with no L0
   penalty. Reuse the existing normalized Horvitz–Thompson `w/q` baseline.
   The stretch multiplier remains 10 **relative to that inclusion-adjusted
   baseline**; this is the shared exact-k refit's contract, not a claim that
   sparse weights remain within 10 times the unexpanded pool-row weights.
   The manifest names the reference explicitly. Its empirical suitability for
   UK release remains to be assessed alongside the size scorecard.
5. Restore the prepared carrier and export only selected linked entities.
   Re-run the existing local gate battery on the compact frame. Each holdout
   fold independently reruns selection on training targets; held targets do
   not inform the prior or protected set.
6. Record requested/realized counts, pool positions, inclusion probabilities,
   protected count, seed, learned penalty, dense/refit loss and target change,
   the gate reports, and ordinary byte-pinned output metadata.

## Running a candidate

Use the inputs and environment from the existing
[UK dense assembly runbook](uk-dense-release-assembly-runbook-762.md).
Pass the same pinned source arguments to the existing driver and add:

```bash
uv run python tools/build_uk_rowwise_candidate.py \
  --input-h5 "$UK_SPINE_H5" --input-sha256 "$UK_SPINE_SHA256" \
  --ladder "$UK_LADDER_NPZ" --ladder-sha256 "$UK_LADDER_SHA256" \
  --ledger-facts "$UK_LEDGER_FACTS" \
  --ledger-facts-sha256 "$UK_LEDGER_FACTS_SHA256" \
  --ledger-manifest-sha256 "$UK_LEDGER_MANIFEST_SHA256" \
  --dataset-households 50000 --seed 42 --out out/uk-k50000
```

Repeat with another positive household count and a fresh output directory to
compare sizes. These are requested counts, not certified presets. Omit
`--dataset-households` to retain the existing dense path. Add `--dry-run` to
inspect input binding and parameters without solving or writing a candidate.
Never reduce `--n-clones` to request a smaller output. Full builds still need
the dense build's peak memory and add L0/refit work; the reduction is in the
exported dataset's storage and downstream loading/simulation footprint.

## Certification and publication still required

The implementation produces **candidates**, not a new certified UK default.
A size request refuses `--release-candidate`, and its manifest records
`releasable=false` even if the diagnostic gate run passes. The dense assembler
must not interpret a compact candidate as the already reviewed dense line.
The existing registry, production pointers and pe.py default are unchanged.

Before any size can be promoted:

1. Run the licensed full-input build, retain all gate failures and measure
   local ESS and fit. A nominal 50k size is not guaranteed to clear the floors
   that motivated K=15.
2. Run #355's matched sound-comparison protocol and the referenced promotion
   scorecard, including reform/distributional validation and untargeted bases.
   Calibration loss alone is not a certificate. The included same-target loss
   comparison is a diagnostic, not a substitute for those protocols.
3. Adjudicate any new size-specific acceptance decisions, including the
   inclusion-adjusted stretch reference. A national-only product would need
   its own explicit scope; this implementation does not downgrade local claims.
4. Add size-specific certified release identities, assembly contracts and
   downstream bundle entries against that evidence; publication remains the
   repository's deliberate human step.

Accordingly, this increment does not close #355's default-flip requirement.
Synthetic CI checks prove code behavior and artifact structure; they do not
establish licensed-data fit, storage measurements, or release eligibility.
