# microcosm#686 whole-spine parity and swap acceptance — measurement receipts

Receipts for the E10 increment (WS-E #145, epic #665). Every value below is a
digest, a count of columns, an unweighted share, or a relative delta per
CD171 §5.2.1 — no unit-record values, no absolute weighted totals, no small
cells. The licensed weighted register itself stays uncommitted under the UKDS
EUL (#609); only its digest and relative movement appear here.

Licensed evidence directory: `data/ukds/acceptance/686-spine-swap/`.

---

## R0 — incumbent reference re-pin, 1.56.14 → 1.56.16

### Why the pin moved

The frozen parity instruments were pinned at policyengine-uk-data **1.56.14**
(`enhanced_frs_2024_25.h5`, HF revision `a2039519…`, sha256 `97a07f9c…`,
126,579,434 B). That artifact carries **policyengine-uk-data#461**, fixed
upstream in `6591b70` and released as **1.56.16**:

> From the 2024-25 FRS release the raw tables are no longer ordered by sernum.
> The household table is already sorted for this reason, but the benunit table
> was not, so every benunit-level variable (including benunit_id itself) was
> assigned to the wrong benefit unit relative to the model's sorted entity
> order.

The parity screen compares **unweighted nonzero shares**, which are invariant
under a row permutation. It is therefore structurally incapable of seeing this
defect. Measured directly on the two artifacts: `is_married` reports
**0.256587 in both**, and its value multiset is identical across the pins,
while `benunit_id` is *not* sorted ascending at 1.56.14 and *is* at 1.56.16.
Signing whole-spine parity against 1.56.14 would have frozen the upstream
defect into the contract as though it were correct.

**Adjudication (María, 2026-08-22): re-pin to 1.56.16.**

### The new pin

| field | 1.56.14 (was) | 1.56.16 (now) |
|---|---|---|
| HF revision | `a2039519d3b92aecc06c66dfd175cb46ac24cada` | `a9e52499b6a6cca100a5ce4f36ca27b2e8a213df` |
| sha256 | `97a07f9c…e37d6c3e` | `e433e532…19a68712` |
| size_bytes | 126,579,434 | 126,553,300 |

Corroborated three independent ways: the HF LFS `sha256` metadata at the tagged
revision, the repo's own `releases/1.56.16/release_manifest.json`, and a local
hash of the downloaded blob — all agree. The reference JSON now also records
`source.version` so the release line is self-describing rather than inferable
only from the revision hash.

`uk/frs_release.json` is deliberately **not** re-pinned: its `a2039519`
revision names the raw UKDS FRS **zip** (`frs_2024_25.zip`, sha
`05dd0069…`, 46,637,202 B), a different artifact that did not change. HF
revisions are immutable, so that pin remains exactly valid.

### Structural verification of the new artifact

| check | 1.56.14 | 1.56.16 | verdict |
|---|---|---|---|
| store keys | person/benunit/household/time_period | same | equal |
| person rows | 113,617 | 113,617 | equal |
| benunit rows | 61,223 | 61,223 | equal |
| household rows | 52,846 | 52,846 | equal |
| column counts | 99 / 11 / 66 | 99 / 11 / 66 | equal, same order |
| `clone_index` | {0} | {0} | **pre-clone confirmed** |
| `household_is_spi_synthetic` | 20,089 | 20,089 | equal |
| `household_is_capital_gains_clone` | 26,421 | 26,421 | equal |
| `household_is_cgt_band_donor` | 270 | 270 | equal |
| `benunit_id` sorted ascending | **False** | **True** | the #461 fix |

The record-count identity holds unchanged at the new pin:
(16,288 raw FRS + 10,000 SPI) × 2 capital-gains clone + 270 CGT band donors
= 52,846 households. `clone_index` is uniformly 0 in both artifacts — the
release workflow builds with `PE_UK_DATA_OA_CLONES=1`, so the published
enhanced-FRS is pre-clone and compares row-for-row with the microcosm spine
grain (the #688 staging-input ruling).

### Defect footprint in the incumbent

Columns whose values differ between the two artifacts:

| entity | differing / total | note |
|---|---|---|
| benunit | **11 / 11** | the whole table; `benunit_id` and `is_married` are pure permutations (identical multisets), the nine `would_claim_*`/opt-out columns are identity-keyed draws re-drawn on corrected keys |
| person | 1 / 99 | `student_loan_balance` |
| household | 34 / 66 | imputed wealth/consumption surfaces (benunit-derived predictors) plus `household_weight` and the post-calibration scalers |

So the misassignment is not confined to the benunit table: it propagates into
the household imputations through benunit-derived predictors, and the
household movement additionally mixes in the 1.56.15 recalibration.

### Microcosm is not exposed

`uk_runtime/frs_spine.py:352` reads
`frs["benunit"].sort_values("benunit_id").reset_index(drop=True)` and line 394
takes `is_married` from that sorted frame. This has been the code since the
original ingest commit `c199347b` (2026-08-14) — four days *before* the
upstream fix (2026-08-18). The port independently got the sorted-id entity
declaration right and never carried the defect, so **no microcosm-side fix is
required**; W2 closes as "already correct, recorded here".

### Reference-side movement (unweighted shares, committed instrument)

Regenerated `uk/efrs_parity_reference.json`: **145 populated input layers**
before and after, no columns added or removed, `entity_stats` identical,
engine identical (`policyengine-uk` 2.89.0, 223 input variables, 866
engine-known persisted).

**39 of 145 shares moved; the largest absolute move is 0.004542.** Nothing
moved beyond ±0.02, so the ±0.02 parity screen — and the #723 classification
of 27 columns beyond it — is undisturbed by the re-pin.

| entity | moved / total | largest \|Δ\| |
|---|---|---|
| household | 29 / 51 | 0.004542 (`transport_consumption`) |
| benunit | 9 / 10 | 0.001911 (`would_claim_extended_childcare`) |
| person | 1 / 84 | 0.000123 (`student_loan_balance`) |

`is_married` is absent from that list precisely because its share is
unchanged — the permutation-blindness restated as a measurement. It is worth
being explicit that this was live rather than hypothetical: `is_married` is
one of the 145 columns the parity instrument compares, so a whole-spine parity
run against the 1.56.14 reference would have compared it, found 0.256587 on
both sides, and reported agreement while every value sat on the wrong benefit
unit. A share screen cannot detect a permutation; that is what the identity
receipts and the at-pin row-level comparator are for, and it is the reason the
pin had to move rather than the divergence being signed.

### Licensed weighted register

Re-emitted with `build_uk_efrs_parity_reference.py --emit-weighted-totals` to
`data/ukds/acceptance/686-spine-swap/uk_input_mass_reference_2024_25_v1_56_16.json`
(uncommitted, UKDS-derived). **131 weighted input totals**, same column
surface as the 1.56.14 register.

| register | evidence sha256 |
|---|---|
| 1.56.14 | `e70a45387c6adc13df5d7eb7da3c2cada7972a2f293a9238c8c29c9e885e4659` |
| 1.56.16 | `fd41cb5f6cf6c4ef812320f21d1942173d49ce6f8725b21fbc9d9ca5423d298c` |

The 1.56.14 digest reproduces the value previously committed as
`UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256`, which validates the computation
before the new digest replaces it.

Relative movement: **all 128 comparable columns move**, versus 39 of 145 on
the unweighted side. 41 columns exceed 2%, 20 exceed 5%, 12 exceed 10%, 6
exceed 25%. This is dominated by `household_weight`, not by the benunit fix:
1.56.15 changed the Universal Credit caseload targets (uk-data `cbd5ae1`) and
the incumbent's calibration re-solves with **unseeded** `torch.rand_like`
dropout, so its weight column is not reproducible run-to-run in the first
place. That is the already-signed register-realization class (E4/E5), and the
movement stays far inside `uk_input_mass_parity`'s gross-mass fence
(`relative_tolerance` 4.5218…), which the re-pin does not change.

### Pins moved in this change

| pin | file |
|---|---|
| `SOURCE_REVISION` / `SOURCE_SHA256` / `SOURCE_SIZE_BYTES` / new `SOURCE_VERSION` | `tools/build_uk_efrs_parity_reference.py` |
| reference `source` block (regenerated) | `uk/efrs_parity_reference.json` |
| `uk_input_mass_parity.reference_registry` identity + `totals_sha256` | `uk/gates.json` |
| reference identity block | `uk/release_input_coverage_manifest.json` |
| `_UK_INPUT_MASS_REFERENCE_DESCRIPTOR`, `UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256` | `uk_runtime/weighted_integrity.py` |
| sha/revision/size literals | `test_uk_parity_reference.py`, `test_uk_terminal_gates.py`, `test_uk_weighted_integrity.py` |

Gate-battery mirrors re-cut in the same change (the lockstep at
`test_gate_battery_contract_pins.py`), computed from the live producers:

| mirror | was | now |
|---|---|---|
| `_UK_GATE_BATTERY_POLICY_SHA256` | `5cb072a0…` | `623f340d…` |
| `_UK_GATE_BATTERY_GATES_MANIFEST_SHA256` | `c5123517…` | `f2cc2af4…` |
| `_UK_GATE_BATTERY_SPEC_FINGERPRINT` | `23cf63b6…` | `3601b4c7…` |
| `_UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256` | `806f46de…` | `16093e86…` |

`_UK_GATE_BATTERY_DEGENERATE_EVIDENCE_SHA256` is unchanged — the degenerate
register did not move. Mirrored in both `microcosm-data/src/.../contract.py`
and its schema-3-style local copies in `microcosm-data/tests/test_contract.py`.

---

## R1 — Scottish water and sewerage charges (#736 item 13)

### What the vintage changed

FRS 2024-25 retired two cells and replaced them (SN 9563,
`9563_dv_summary_2425.xlsx`, `9563_frs2425_variable_listing_eul.xlsx`,
`9563_frs202425_changes.xlsx`):

| cell | label | status in 2024-25 |
|---|---|---|
| `CWATAMT` | Wat. Charge: Final value **after discount** | header present, **no data** |
| `CSEWAMT` | Sew. Charge: Final value **after discount** | header present, **no data** |
| `CWATAMT1` | Weeklyised **gross** annual dom. water charge on bill | new DV, Scotland only |
| `CSEWAMT1` | Weeklyised **gross** annual dom. sew. charge on bill | new DV, Scotland only |
| `CWATAMTD` | Deriv Council Tax water charge -Scot (discount applied) | unchanged |

The changes workbook lists `CWATAMT1`/`CSEWAMT1` as "Added as DV in 2425", and
each DV's own note says it was created "as variable was removed from the
dataset for 2024-25". Measured on the tab: `CSEWAMT` and `CWATAMT` are blank
in **all 16,288** households. All five cells sit in the FRS weekly-variables
listing, so `WEEKS_IN_YEAR` still applies; `ORGWATAMT`/`ORGSEWAMT` are
**annual**, not weeklyised (their ratio to the weeklyised DVs is ≈52.18), and
are therefore not interchangeable with them.

### The divergence is the incumbent's

The incumbent computes `np.where(scotland, csewamt + cwatamtd, watsewrt)` and
fills afterwards, so a wholly blank `CSEWAMT` propagates NaN across the
addition and zeroes the charge for every Scottish household. Reproduced
directly on the raw tab:

| formula | nonzero households | nonzero share |
|---|---|---|
| incumbent (`csewamt + cwatamtd`, fill after) | 12,644 | 0.776277 |
| microcosm (per-column fill) | 14,307 | 0.878377 |
| difference | 1,663 (all Scottish) | **+0.102100** |

That reproduces the +0.1009 screen divergence before composition, and
microcosm's 0.878377 matches `shares-a.json → stages.frs_spine`
(0.8783767190569745) to every digit. By country the difference is entirely
Scotland (England 11,483 and Wales 1,161 identical in both; Northern Ireland
zero in both, correctly — NI water is inside the regional rate). The defect
is latent from at least 2023-24, where `CSEWAMT` was already missing for 378
Scottish households (share gap 0.0224 ≡ 378/16,754), and total in 2024-25.
It survives at 1.56.16: the only build-path change in that release was the
benunit sort.

### The level fix

`CWATAMTD` is the water charge alone, so emitting it unaccompanied understated
the Scottish bill. Weighted annual per Scottish household (2,564,102 grossed
households), against England and Wales on `WATSEWRT` at £489.67:

| basis | £/household |
|---|---|
| incumbent (NaN-zeroed) | 0.00 |
| `cwatamtd` alone (microcosm before this change) | 184.50 |
| **`cwatamtd` + `csewamt1` × own discount factor (adopted)** | **395.43** |
| `cwatamt1 + csewamt1` (both gross) | 508.63 |

The adopted basis preserves the semantics of the cells it replaces: the
retired pair was *after discount*, the successors are *gross*, and the FRS
publishes no discounted sewerage counterpart, so the household's own factor
`CWATAMTD / CWATAMT1` carries the discount to the sewerage side of the same
bill. Taking the gross basis instead would silently change what the variable
means; taking `cwatamtd` alone leaves it short by the sewerage component.

Factor domain on the tab — the three cases the helper's tests pin:

| domain | households | treatment |
|---|---|---|
| `CWATAMT1 > 0` | 1,641 | factor observed; range (0.3333, 1.0], mean 0.7529, never > 1 |
| `CWATAMT1 == 0`, `CWATAMTD > 0` | 22 | `CSEWAMT1` is 0 for all 22, so the fallback factor cannot move them |
| all council-tax cells absent | 21 | fall to zero, unchanged |

The gross sewerage-to-water ratio is stable at 1.09–1.25 (median 1.147),
consistent with Scottish Water charging sewerage slightly above water.

**The level fix does not move the nonzero share** (0.878377 either way) — the
same households are charged. So the share divergence stands alone as a signed
incumbent-defect difference, and the level change is a separate signed
deviation from the incumbent.

### Consistency and coverage

Both consumers now call one helper, `scottish_water_and_sewerage_weekly` in
`uk_runtime/frs_spine.py`: the spine's `water_and_sewerage_charges` and the
`frs_council_tax` netting. The incumbent nets a different amount from the
council tax bill than it charges as water (its netting fills per column, its
charge fills after the addition), which is an internal inconsistency the
shared helper removes by construction.

Two coverage gaps that let this survive, both now closed in PR CI: the unit
fixtures supplied a non-missing `CSEWAMT` and so never exercised the vintage's
actual content, and the #723 raw-vintage audit compares header **sets**, which
cannot see a retained-but-empty column (`consumed_column_regressions: {}`).
The audit is an ad-hoc licensed script rather than committed tooling, so the
durable guard is the regression test
`test_retired_cells_cannot_reintroduce_the_incumbent_zeroing`, which asserts
that a blank retired cell and an absent one give the same non-zero answer.

The retired cells are no longer read anywhere in the runtime.

---

## R2 — parity instrument, first end-to-end run

Not an acceptance run: the E8 spine artifact predates both the re-pin and the
Scottish water fix, and the signed register is deliberately seeded with only
the two adjudications made so far. The run exists to prove the instrument
behaves as specified on real artifacts before the L0–L3 ladder depends on it.

Candidate extracted from `data/ukds/acceptance/e8/spine-a.h5` with
`build_uk_efrs_parity_reference.py --candidate-h5 --emit-candidate-json`
(144 candidate input layers), diffed with `verify_uk_spine_parity.py`.
Evidence: `data/ukds/acceptance/686-spine-swap/`
(`candidate_extraction_e8_spine_a.json`, `parity_receipt_e8_spine_a.json`).

**Verdict `defect`, exit 1** — correct for a deliberately under-seeded
register, and the state the L2 loop starts from.

| surface | result |
|---|---|
| entity counts | household 52,846 = 52,846 (identity holds); person 113,617 vs 113,649 and benunit 61,223 vs 61,211 differ — the known E8 donor-composition outcome, not yet transcribed into the register |
| shares | 142 compared, 113 differing, 3 missing in candidate, 2 extra |
| unsigned | 119 |

The three columns missing on the candidate side are `free_school_meals`,
`free_school_fruit_veg` and `healthy_start_vouchers` — the E9 derived-benefit
class the #723 receipt already named, unchanged by the re-pin.

Two behaviours worth recording, because they are the ones the swap decision
depends on:

* `water_and_sewerage_charges` measured **+0.100897** (0.776937 → 0.877834)
  and bound to `scottish-water-incumbent-nan-zeroing`, so it is reported as a
  *signed* difference and excluded from the unsigned list. The share
  reproduces the divergence #736 recorded, now against the re-pinned
  reference, and this candidate predates the level fix.
* `scottish-water-sewerage-successor-level` correctly appears under
  `unused_ids`: it signs `weighted_totals`, and this run supplied no weighted
  registers. Under `--strict` that would fail the run, which is the intended
  swap-acceptance behaviour — a register entry that matches nothing is either
  stale or the run is incomplete, and both deserve to stop the swap.

---

## R3 — the spine, rebuilt; and the L2 adjudication queue

### L0 ladder

Built from the raw licensed tabs on this branch (re-pin + water fix), FRS
2024-25, all 24 stages, every attempt landing a Logbook row with disposition
`iterating`:

| rung | households (post-stack) | verdict |
|---|---|---|
| smoke `f001` | 794 | built clean |
| dev `f010` | 5,526 | built clean |
| full `1.0` (twin A) | 52,846 | built clean |

**The record-count identity closes exactly at full scale:**
(16,288 raw FRS + 10,000 SPI) × 2 capital-gains clone + 270 CGT band donors
= **52,846** households, matching the pinned incumbent to the row.
Persons 113,649 and benefit units 61,211 against the incumbent's 113,617 and
61,223 — the standing E8 donor-composition outcome, unchanged by this rebuild.

The Scottish water fix reproduces at every scale: the `frs_spine` share is
`0.8783767190569745` in all three builds (it is measured before sampling, so
the rungs agree by construction), and the weighted Scottish charge lands at
£390.80 / £375.08 / £405.34 across smoke / dev / full against roughly £185
under the retired mapping.

### Parity against the re-pinned reference

`verify_uk_spine_parity.py` over the full twin-A extraction (144 candidate
input layers): 142 columns compared, 113 differing, 3 missing, 2 extra,
**26 beyond ±0.02**. For comparison the #723 screen found 27 beyond the band
against the 1.56.14 reference, which is the predicted outcome of a re-pin
where no reference share moved by more than 0.0046.

### Attribution — and a correction to how it is done

Divergences are attributed to **the stage whose values survive**, which is the
last stage to either produce *or rewrite* the column. Attributing by producing
stage alone is wrong and produces false findings: `savings_interest_income`
and `tax_free_savings_income` both originate in `frs_spine` and are then
rewritten by `hmrc_spi_income_spine`, so a naive attribution reports them as
raw-mapping divergences outside every signed class — the same signature that
made the water defect real. They are E7, not new findings. The distinction
matters precisely because the raw-mapping signature is the one that indicates
a genuine defect rather than a method difference.

With rewrites folded in, **every beyond-band divergence falls in an
established class**, and the only raw-mapping one is already signed:

| class | count | columns |
|---|---|---|
| E6 consumption | 15 | bus_subsidy_spending, dfe_education_spending, restaurants_and_hotels, petrol, education, household_furnishings, electricity, miscellaneous, communication, alcohol_and_tobacco, gas, domestic_energy, diesel, transport, health |
| E5 wealth QRF | 6 | savings, property_wealth, corporate_wealth, other_residential_property_value, main_residence_value, student_loan_balance |
| E7 SPI channel | 3 | tax_free_savings_income, employer_pension_contributions, savings_interest_income |
| E8 CGT/salsac/loans | 1 | employee_pension_contributions |
| E2 raw FRS mapping | 1 | water_and_sewerage_charges — **signed** |

Zero divergent columns are unattributable to a stage.

### The queue — pending María's adjudication

The parity verdict is `defect` by construction: the register holds only the two
water adjudications, so all 119 remaining differences are unsigned. That is the
correct starting state, not a failure. Each of the following needs a ruling
before it becomes a register entry; none should be transcribed on its prose
classification alone.

1. **E6 consumption, 15 columns.** The largest deltas in the whole screen
   (bus_subsidy +0.238, dfe_education +0.225). The E6 acceptance established
   ours is donor-faithful and the incumbent collapses zero-inflated targets;
   if that stands, one class entry scoped to these 15 covers them.
2. **E5 wealth QRF, 6 columns.** The standing correlated-rank-draw difference.
   `owned_land` is *not* among them — worth noting given its exclusion expires
   2026-09-20.
3. **E7 SPI channel, 3 columns.** The U14 QRF-surface class at ~38% synthetic
   composition, carried from #717 and still pending adjudication there.
4. **E8, 1 column** — `employee_pension_contributions`, the signed
   conversion-depth difference from #684.
5. **Entity counts.** Persons +32, benefit units −12. Signed at E8 as a
   donor-selection RNG outcome; needs an `entity_counts` register entry, which
   is the one surface where a surface-wide entry is legitimate.
6. **Missing, 3 columns** — `free_school_meals`, `free_school_fruit_veg`,
   `healthy_start_vouchers`: the E9 derived-benefit family. This is a genuine
   coverage gap rather than a method difference, and it is the one item here
   that may argue against swapping rather than for signing.
7. **Extra, 2 columns** — `num_bedrooms` (`frs_spine`) and
   `other_investment_income` (`hmrc_spi_income_spine`). Net-new columns the
   incumbent does not populate; `other_investment_income` is declared by the
   incumbent's own national restoration, so the spine is ahead of the pinned
   artifact rather than behind it.

---

## R4 — determinism, and a stale scope in the identity ladder

### Twin determinism: PASS

Two independent full builds compared with `compare_uk_h5_payload.py`:
**payload-identical** across all four store keys, no differing root
attributes, entity rows equal (52,846 / 61,211 / 113,649). The two files
differ in bytes, which is correct — `HDFStore` stamps object-header write
times, which is exactly why payload comparison rather than byte comparison is
the instrument.

### Identity receipts: the determinism property holds everywhere

| check | `identical_under_permutation` | `matches_stored_columns` |
|---|---|---|
| e4 | **PASS** | fail |
| e5 | **PASS** | fail |
| e6 | **PASS** | fail |
| e8 | **PASS** | **PASS** |

The property these receipts exist to prove — that recomputation is invariant
to row order — holds in all four. What fails is the secondary comparison of
the recomputation against the columns stored in the artifact, and the cause is
in the instrument rather than the spine.

### Why, and the measurement that shows it

`_frs_only_frame` (`tools/verify_uk_identity_stability.py`) scopes the frame
to the survey channel by excluding `household_is_spi_synthetic`. It was
written for the post-#717 artifact, where the SPI channel was the only layer
stacking rows. **E8 added two more**: the capital-gains incidence clone (×2)
and the 270 CGT band donors. Those rows carry values *copied from their source
rows*, so recomputing an identity-keyed draw for a clone's own household id
legitimately disagrees with the stored value — the stored value was never
drawn for that id.

Measured on `spine-a.h5`:

| scope | households |
|---|---|
| all rows | 52,846 |
| `~spi_synthetic` — what the tool currently uses | 32,761 |
| `~spi_synthetic & ~capital_gains_clone` | 16,381 |
| `~spi_synthetic & ~capital_gains_clone & ~cgt_band_donor` | **16,288** |

16,288 is the raw FRS household count, and it is exactly the scope the #723
receipts ran at (`entity_row_counts` 16,288 / 18,850 / 34,966 with
`matches_stored_columns: true`). The #723 spine predates E8, so its
`~spi_synthetic` scope *was* the raw-FRS scope; E8's stacking silently widened
it. e8's own receipt passes because it recomputes the clone and donor logic
explicitly rather than assuming unstacked rows.

The e4 mismatch list is consistent with this reading throughout: it is the
identity-keyed draw columns (`would_claim_*`, `household_owns_tv`,
`would_evade_tv_licence_fee`, `property_purchased`, `brma`, …), which are
precisely the columns whose values a clone inherits rather than draws. e5 and
e6 report the failure with an empty mismatch map, which is its own small
defect in the receipt's reporting — a fail with nothing named is not
actionable evidence, and should name what disagreed.

**Disposition: fix the instrument, do not sign this.** The scope must exclude
every stacked layer, not just the SPI channel, and the fix belongs with the
e7 receipt work (W4) since both are ladder maintenance. Until then the e4/e5/e6
`matches_stored_columns` results carry no information about the spine, and the
L1 leg of the gate is not satisfied — the receipts must be re-run after the
fix rather than accepted as-is.

### Carried consequence

The re-pin does not by itself re-validate the #723 acceptance screen: the
27-columns-beyond-±0.02 classification was measured against the 1.56.14
reference. Because no reference share moved by more than 0.0046, that
classification is expected to survive intact, but it is re-measured against
the new reference in the L2 whole-spine parity loop rather than assumed.
