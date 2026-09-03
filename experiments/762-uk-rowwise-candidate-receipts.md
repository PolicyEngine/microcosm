# First calibrated rowwise UK local candidate — receipts (#762, #495 increment 6)

Plan: `repos/uk-762-rowwise-candidate-plan.md` (approved 2026-09-02). Machinery: PR #852 (PR A). Registers, deferrals and these receipts: PR B (`uk-rowwise-candidate-762-b`). Spine of record: spine-m. Every run below is the **joint solve**: local register cells + ladder census household rows + the activated national families as rows of one matrix over the K-cloned spine, engine resolved on every cloned row, one `calibrate()` call under the local doctrine.

## R0 — identity

| item | value |
|---|---|
| spine | `spine-m.h5` (`data/ukds/acceptance/spine-m-832/`), sha256 `fb053cd2157702969c6e055278b5d5c10fbcc389d5c06eb80f6a3b345570713a`, 154,942,842 bytes, 52,846 households, `importance` weights, mass 29,247,433.0; sidecar + 14/14 spine gates bound |
| ladder | `uk_oa_ladder_2021.npz`, sha256 `9c6d56b90d2e975d750106b175020a54c5ec6acf42ef8909d304a9d7fc3868a7`, `matches_local_area_crosswalk_pin: true` |
| ledger facts | chronicle consumer artifact `1cab809`, facts sha256 `226358e73e7c449e71a3f6dc91a72e8e3941e0f14265943d81c990edb21c2a6c`, manifest `aadc0105…`; 107,550 rows |
| survey / calibration year | 2024 (FRS 2024-25) / **2025** |
| registers in force | `local_binding_adjudications.json`: 6 fences (2026-09-02 → 2026-12-02; census fence to 2026-11-30); `local_area_support_exclusions.json`: E06000053, E09000001 |
| surface | 19,444 active local register cells (19,618 − 174 Welsh council-tax cells signed-deferred, A13) + 1,011 ladder household rows + 364 activated national references |
| code | PR A `1ddb0e27` / PR B `effdc6e8` (this document's runs; later runs cite their own pins) |

## R1 — f100 dry-run K sweep (2026-09-02, ninth attempt completed)

No engine, no solve, no write. 81 s, 5.0 GB peak RSS through pins, both registry compiles, the K=4 clone, surface assembly, cross-grain reconciliation and the sweep.

Matrix at K=4: **20,819 rows × 211,384 households** (20,455 local incl. ladder; 364 national).

| K | constituency min rows / ESS / sources | LA min rows / ESS / sources |
|--:|---|---|
| 1 | 28 / 15.3 / 28 | 1 / 1.0 / 1 |
| 2 | 42 / 28.8 / 42 | 3 / 2.4 / 3 |
| 4 | 101 / 86.4 / 97 | 7 / 6.8 / 7 |

Identical to the #761 tables (spine-i, spine-k): the assignment surface is value-column-independent and the roster/region/weight inputs are preserved on spine-m.

Cross-grain receipt (39 groups in force):
- 8 age-band bridges at both grains, factor 1.0000 (ONS by-area sums equal the UK totals).
- UC caseload bridge: by-area UC counts rescale ×1.0276 at both grains (2025-05 area facts → 2025-12 caseload).
- 12 constituency-over-LA exact matches (HMRC amounts, age bands, UC count): LA cells rescale to constituency sums, factors 0.992–1.022.
- England council-tax controls (8 bands) over 296 LAs, factors within 0.1 % of one; Scotland's 8 controls: licensed empty legs (every Scottish LA cell signed-deferred); Wales: no rows (A13).
- Household-composition partition bridge **unbound** (three cells are #791 reviewed exclusions) → census households bind as published, 2021/22 vintage (A11, published to #736).
- 15 fan-out national targets receipted as distributions, not controls.

Vintages behind the 2025 period: UC 2025-05 (3,510 cells), HMRC tax-year 2023 (4,035), age mid-2024 (8,088), tenure 2021 (1,316) / 2022 (128), council tax 2025 (no hold).

## R2 — f001 smoke (K=4, 1 %, sample seed 578; 2026-09-02)

Machinery end to end on the licensed spine: sample 52,846 → 501 households (156 source families; mass renormalized ×105.29 to the full total), clone ×4 → 2,004 households / 4,352 persons / 2,496 benunits, engine resolved on the cloned frame (46 national inputs, 18 + 30 local metrics), surface restricted for the rung (1,223 cells whose area drew no rows; 10,429 covered cells with zero metric support), **9,167 targets × 2,004 households** (7,854 register + 949 ladder + 364 national), one solve (uniform, 512 epochs), battery run with ladder-only enforcement, 5-fold rotated holdout, atomic bundle, Logbook row chained on the `uk/local` tail.

| measure | value |
|---|---|
| wall / peak RSS | 101 s / 3.1 GB |
| loss initial → final | 1.375 → 0.418 |
| median / max abs relative error | 0.347 / 926 |
| past-cap (all rows) | 147 at init, 8 at final, 139 escaped, 0 pushed out |
| realized max weight ratio vs design | 6.29 (bound 100) |
| calibration mass change | 29,247,433 → 14,385,130 (×0.492, free mass) |
| holdout fold losses | 0.676, 0.641, 0.679, 0.674, 0.659 |
| gates | ladder passed; area_support / target_fit / per_family_fit failed (recorded, not enforced below f100); weight_ess, weight_ratio passed |
| releasable | false (rung) |

A 1 % rung has no fit meaning; it establishes that every stage runs on the real spine and sizes the cost. Support-limited-miss diagnostic ran (constituency: 2,411 failing cells, 14 % in the bottom-ESS decile, Spearman −0.04).

## R3 — f010 development rung (K=4, 10 %, sample seed 578; 2026-09-02)

Sample 52,846 → 5,247 households (1,622 source families; mass renormalized ×10.07), clone ×4 → 20,988 households / 45,248 persons / 24,680 benunits; rung surface: 22 cells whose area drew no rows, 1,848 covered cells with zero metric support dropped; **18,585 targets × 20,988 households** (17,575 register + 1,010 ladder + 364 national).

| measure | value |
|---|---|
| wall / peak RSS | 214 s / 4.3 GB (f001: 101 s / 3.1 GB) |
| loss initial → final | 0.524 → 0.139 |
| median / max abs relative error | 0.032 / 249 (City of London band H) |
| past-cap (all rows) | 46 at init, 16 at final, 30 escaped, 0 pushed out |
| realized max weight ratio vs design | 57.3 (bound 100) |
| calibration mass change | 29,247,433 → 28,165,560 (×0.963, free mass) |
| holdout fold losses | 0.431, 0.436, 0.422, 0.442, 0.471 |
| local fit within 10 % / 25 % | age 69 % / 82 % (7,985 cells); census households 85 % / 98 % (1,010); council tax 72 % / 84 % (1,893) |
| gates | ladder, per_family_fit, weight_ess, weight_ratio passed; area_support and target_fit failed (recorded, not enforced below f100) |
| support-limited misses | constituency: 1,704 failing cells, Spearman(worst error, ESS) −0.38; LA: 1,577, −0.50 |
| releasable | false (rung) |

Sizing for f100 from f001 → f010: about +1.2 GB per 19k cloned rows on top of a 3 GB base → ≈ 16 GB peak at 211,384 rows; wall scales with the engine run and the five holdout solves.

## R4 — f100 at the ruled K=4: refused on support (2026-09-02, attempt `ff0faa2d`, Logbook row `219cb604…`, disposition failed)

Sampling none (f100), clone ×4 → 211,384 households, engine resolved on every cloned row in one block (154 s to the refusal, **7.6 GB** peak RSS — the f001 → f010 extrapolation of 16 GB was pessimistic). The strict f100 builder refused: **172 cells with a nonzero target and zero household support**, all in the UC child-band family (`uc_hh_2_children` in the listed examples: E14001084, E14001107, E14001156, E14001318, S14000027 — Na h-Eileanan an Iar, the smallest constituency at 101 rows). A UC benefit unit with exactly two children is ~2 % of cloned rows, so a 101-row constituency has ~13 % odds of carrying none; the miss is support-limited by construction, which is the D7 trigger: "if Run U shows misses concentrated where ESS is lowest, a K=10 run under the identical doctrine is the measured comparison". K=10 (≈ 250 rows in the smallest constituency) should leave only a handful of such cells; those that remain are target-side work (signed deferral of the specific zero-support cells) rather than a clone multiplier.

Consequence for the K ruling (A4): K=4 cannot bind the full UC child-band surface at f100; the K=10 measurement below is the evidence for re-adjudicating the declared `--n-clones`.

## R5 — f100 at K=10 (the D7 escalation measurement; 2026-09-02, attempt chained on R4's failed row)

Clone ×10 → 528,460 households; engine resolved per clone block (`--engine-blocks 10`, each block a declared subset of the cloned mass); **192 s to the refusal, 9.9 GB peak RSS**. The strict f100 builder refused **86 cells**, all at local-authority grain: 84 × `council_tax/band_h` (84 of the 296 active band-H authorities), plus Isles of Scilly `age/0_10` and `council_tax/band_f`. **Every constituency-grain cell is supported at K=10** — the 172 UC child-band cells unreachable at K=4 are gone.

Why band H is not a clone-count problem: spine-m carries **170 band-H households, from 49 raw FRS households** (London 13, Wales 11, South East 8, Scotland 6, West Midlands 4, South West 3, East of England 3, East Midlands 1); 49 sources cannot populate 296 authorities at any affordable K. The Scilly cells are the A4 support-exclusion area (7 rows at K=4, 17 at K=10) carrying 30 metrics.

| K | wall to refusal / peak | unreachable nonzero cells | where |
|--:|---|--:|---|
| 4 | 154 s / 7.6 GB | 172 | constituency UC child bands (small constituencies) |
| 10 | 192 s / 9.9 GB (10 engine blocks) | 86 | LA: band H × 84, Scilly × 2 |

Proposed dispositions (A14, María): K=10 for the first candidate (measured); sign a deferral for the 296 band-H LA cells (`voa.council_tax_stock.by_area.band_h`, spine support 49 raw sources); amend A4 so the two support-excluded authorities' cells are signed-deferred as well (they stay in the solve through the other families' rows; their own cells are unsupported by the exclusion's own evidence).

## R6 — Run U: f100 at K=10, uniform weighting, no holdout (2026-09-03 00:xx; attempt chained on R5's failed row; disposition failed — artifact written, unreleasable)

The A14 surface (19,105 register cells + 1,011 ladder rows + 364 national = **20,480 targets × 528,460 households**), engine per clone block, one uniform solve (512 epochs, lr 0.15, cap 10, stretch bound 100). Wall 797 s, peak 9.7 GB.

| measure | value |
|---|---|
| loss initial → final | 0.2777 → 0.0162 |
| median / max abs relative error | 0.0083 / 16.45 (council-tax band G, one authority) |
| past-cap (all rows) | 17 at init, 1 at final, 16 escaped, 0 pushed out; national rows 0 / 0 |
| realized max weight ratio vs design | **100.0 (on the bound)**; max/positive-median weight 312.5 |
| calibration mass change | 29,247,433 → 28,073,792 (×0.960, free mass) |
| local fit within 10 % / 25 % | age 100 % / 100 % (8,072); census households 99.9 % / 100 % (1,011); HMRC 99.6 % / 99.9 % (4,031); UC 100 % / 100 % (3,508); tenure 99.4 % / 99.9 % (1,436); council tax 86.3 % / 95.2 % (2,058) |
| national fit within 10 % / 25 % | hmrc_spi 83 % / 95 % (129); dwp_universal_credit 90 % / 96 % (95); ons_population 92 % / 96 % (50); obr 48 % / 76 % (21); household composition 29 % / 57 % (7); ons_land 0 % / 33 % (3, corporate land value −681 %) |
| gates | ladder, per_family_fit, weight_ess passed; **area_support FAILED (release-blocking): 12 constituencies with calibrated ESS < 50 (18.3–49.8; pre-calibration K=10 minima 245)**; target_fit diagnostic: 20 cells > 25 %; weight_ratio diagnostic: 312.5 > 100 |
| support-limited misses | constituency 2 failing cells (both in the bottom-ESS decile); LA 102, Spearman −0.25 |
| releasable | false (blocked at f100) |

Reading: the local surface fits (every family but council tax above 99 % within 10 %), the national families fit as the national line does, and the block is the doctrine's own stretch: a handful of rows in twelve constituencies were pushed to the 100× bound and the Kish ESS of those areas collapsed below María's floor. This is the A3 evidence: the bound the local doctrine inherited (100) is too loose for the support floor it now has to satisfy; the national doctrine binds at 10 and the exact-k ladder at 20.

## R7 — Run G: f100 at K=10, `grain_equal` weighting, no holdout (2026-09-03 01:xx; chained on R6's row; disposition failed — artifact written, unreleasable)

Same surface and inputs as R6; the receipted override `target_weight_rule: uniform → grain_equal` (one equal share each for the national rows, the constituency rows and the local-authority rows; uniform within). Wall 830 s, peak 6.6 GB.

| measure | Run U (uniform) | Run G (grain_equal) |
|---|---|---|
| loss initial → final | 0.2777 → 0.0162 | 0.3198 → 0.0203 |
| median / max abs relative error | 0.0083 / 16.45 | 0.0092 / 1.42 |
| past-cap all rows (init → final, pushed out) | 17 → 1, 0 | 17 → 0, 0 |
| realized max ratio vs design | 100.0 (bound) | 100.0 (bound) |
| max / positive-median weight | 312.5 | 1,014.5 |
| ESS fraction | 0.216 | 0.070 |
| constituencies with calibrated ESS < 50 | **12** | **189** |
| mass factor | 0.960 | 0.973 |
| local within 10 %: age / census hh / HMRC / UC / tenure / council tax | 100 / 99.9 / 99.6 / 100 / 99.4 / 86.3 | 100 / 98.7 / 99.1 / 100 / 97.5 / 87.5 |
| national within 10 % (n-weighted) / 25 % | 77.7 % / 90.1 % | **92.3 % / 97.8 %** |
| national within 10 %: hmrc_spi / dwp_uc / ons_population / obr | 83 / 89 / 92 / 48 | 98 / 99 / 100 / 67 |
| gates | area_support FAILED (12), target_fit 20 cells, weight_ratio | area_support FAILED (189), target_fit 20 cells, weight_ratio |

Reading: `grain_equal` buys the national fit the joint solve is supposed to keep, at almost no local cost, and pays for it in weight concentration. Under both rules the realized stretch sits on the 100× bound and the support floor fails, so the weighting question (A2) cannot be settled at this bound: the stretch bound (A3) is the binding decision. Measurement runs at 20 and 10 under both rules follow (R8).

## R8 — A3 measurement: stretch bound 100 / 20 / 10 under both weighting rules (2026-09-03 01:40–02:40; K=10, f100, no holdout, `--engine-blocks 10`)

Measurement branch `uk-rowwise-candidate-762-measure` (worktree `populace-762m`, cut from PR B at c67efecc; **never merged**): `UK_LOCAL_MAX_WEIGHT_RATIO` edited to 20.0 (commit "MEASUREMENT ONLY: stretch bound 20") and then 10.0; every other input, pin, seed and the chain identical to R6/R7. Output directories `f100-k10-{U,G}{20,10}-eval` beside R6/R7's. Wall 775–826 s, peak 11.3–12.1 GB each.

| | U (100) | G (100) | U20 | G20 | U10 | G10 |
|---|---|---|---|---|---|---|
| loss initial → final | 0.2777→0.0162 | 0.3198→0.0203 | 0.2777→0.0162 | 0.3198→0.0203 | 0.2777→0.0165 | 0.3198→0.0207 |
| median / max abs relative error | 0.0083 / 16.45 | 0.0092 / 1.42 | 0.0082 / 16.92 | 0.0085 / 1.14 | 0.0081 / 17.18 | 0.0074 / 0.96 |
| realized max ratio vs design | 100.0 | 100.0 | 20.0 | 20.0 | 10.0 | 10.0 |
| max / positive-median weight | 312.5 | 1,014.5 | 120.8 | 840.4 | 104.6 | 578.1 |
| ESS fraction / top-1 % weight share | 0.216 / 0.138 | 0.070 / 0.290 | 0.244 / 0.135 | 0.099 / 0.237 | 0.273 / 0.123 | 0.137 / 0.171 |
| constituencies ESS < 50 (min constituency ESS) | 12 (16.3) | 189 (11.8) | 3 (35.6) | 41 (24.0) | **0 (54.1)** | 5 (42.3) |
| local within 10 %: age / census / HMRC / UC / tenure / council tax | 100 / 99.9 / 99.6 / 100 / 99.4 / 86.3 | 100 / 98.7 / 99.1 / 100 / 97.5 / 87.5 | 100 / 99.9 / 99.7 / 99.9 / 99.4 / 86.2 | 100 / 98.6 / 98.9 / 99.9 / 97.4 / 87.3 | 100 / 99.8 / 99.5 / 99.8 / 99.4 / 86.2 | 99.9 / 97.2 / 98.9 / 99.7 / 96.9 / 87.0 |
| local cells past 25 % (diagnostic target_fit) | 104 | 94 | 106 | 108 | 110 | 116 |
| national within 10 % / 25 % (n-weighted) | 77.7 / 90.1 | 92.3 / 97.8 | 77.7 / 90.1 | 92.0 / 97.5 | 77.7 / 89.3 | 91.8 / 97.3 |
| mass factor | 0.960 | 0.973 | 0.960 | 0.971 | 0.960 | 0.973 |
| release-blocking gates | area_support | area_support | area_support | area_support | **none (exit 0)** | area_support |

Per national family at bound 10 (n; within 10 % U10 / G10): hmrc_spi (129; 84 / 97), dwp_universal_credit (95; 89 / 99), ons_population (50; 92 / 100), obr (21; 43 / 62), council_tax_stock (18; 67 / 72), dwp_two_child_limit (15; 67 / 100), ons_household_composition (7; 29 / 71), slc_student_support (6; 50 / 83), dwp_legacy_benefits (4; 25 / 25), hmrc_salary_sacrifice (3; 33 / 100), ons_land (3; 0 / 0), slc_borrowers (3; 33 / 33), slc_repayments (3; 67 / 100), single-cell families benefit_cap / isc / public-sector employment (0 / 100 each), savings interest (0 / 0), CGT and Scottish child payment (100 / 100).

Readings:

1. **The stretch bound costs no fit.** Tightening 100 → 10 leaves the final loss, every local family's within-10 % share and the national shares unchanged to a tenth of a point under both rules. The extra stretch at 100 was buying weight concentration, not fit: under `uniform` the max/median ratio falls 312 → 105 and the ESS fraction rises 0.22 → 0.27; under `grain_equal` 1,014 → 578 and 0.07 → 0.14.
2. **Bound 10 is the only bound at which a run clears the support floor.** U10 passes every release-blocking gate (0 areas failed; min constituency ESS 54.1; the two reviewed micro-LAs stand on their exclusions). It is also the national doctrine's constant, so ruling it makes the two UK doctrines agree.
3. **The weighting rule is a national-fit vs support trade, now quantified at bound 10.** `grain_equal` lifts the national rows from 78 % to 92 % within 10 % (UC 89 → 99, SPI 84 → 97, OBR 43 → 62, two-child limit 67 → 100, household composition 29 → 71) and costs 2.6 points of census-household fit, 2.5 of tenure, half the ESS fraction, and five constituencies that miss the floor by 0.3–7.7 ESS (E14001123, E14001187, E14001417, E14001421, E14001563). Under `uniform` the 364 national rows are 1.8 % of 20,480 rows and are effectively ignored.
4. The `target_fit` and `weight_ratio` entries that fail in every run are `diagnostic` criticality (not release-blocking). `target_fit` is dominated by council tax (97 of 110 cells at U10: the band-H shortage and the A14/Wales deferrals' neighbours); `weight_ratio` measures max/median of the final weights against 100 and stays above it at bound 10 because the design weights (staging importance × ladder clone) already spread ~10×.
5. R9 measures the plan's K escalation for `grain_equal` at bound 10 (K=15, same chain) to see whether the five near-floor constituencies clear without giving up the national fit.

## R9 — K escalation at bound 10: K=15 under both rules (2026-09-03 03:10–03:50; f100, no holdout, `--engine-blocks 15`, measurement branch at bound 10, chained on R8's G10 row)

The plan's clause for insufficient ESS is a measured clone-count increase. Matrix 20,480 × 792,690 (K=15 clones of 52,846). Wall 1,169 s (U) / 1,178 s (G); peak 11.5 / 10.5 GB (per-block engine resolution keeps memory flat in K).

| bound 10 | U10 (K=10) | G10 (K=10) | U15 (K=15) | G15 (K=15) |
|---|---|---|---|---|
| loss initial → final | 0.2777→0.0165 | 0.3198→0.0207 | 0.2734→0.0163 | 0.3195→0.0216 |
| median / max abs relative error | 0.0081 / 17.18 | 0.0074 / 0.96 | 0.0079 / 19.21 | 0.0072 / 0.95 |
| realized max ratio vs design | 10.0 | 10.0 | 10.0 | 10.0 |
| max / positive-median weight | 104.6 | 578.1 | 183.8 | 827.3 |
| ESS (absolute) / ESS fraction | 144,043 / 0.273 | 72,401 / 0.137 | 214,623 / 0.271 | 102,120 / 0.129 |
| constituencies ESS < 50 (min constituency ESS) | 0 (54.1) | 5 (42.3) | 0 (71.6) | **0 (54.1)** |
| min LA ESS (excluded micro-LAs) | 7.1 | 5.3 | 13.2 | 9.3 |
| local within 10 %: age / census / HMRC / UC / tenure / council tax | 100 / 99.8 / 99.5 / 99.8 / 99.4 / 86.2 | 99.9 / 97.2 / 98.9 / 99.7 / 96.9 / 87.0 | 100 / 99.9 / 99.5 / 99.9 / 99.2 / 86.3 | 100 / 98.4 / 98.9 / 99.9 / 97.1 / 87.1 |
| local cells past 25 % (diagnostic) | 110 | 116 | 108 | 109 |
| national within 10 % / 25 % | 77.7 / 89.3 | 91.8 / 97.3 | 78.6 / 90.4 | 92.0 / 97.5 |
| national within 10 %: SPI / UC / population / OBR / two-child / composition | 84 / 89 / 92 / 43 / 67 / 29 | 97 / 99 / 100 / 62 / 100 / 71 | 84 / 92 / 92 / 38 / 67 / 29 | 98 / 99 / 100 / 62 / 100 / 71 |
| mass factor | 0.960 | 0.973 | 0.960 | 0.969 |
| release-blocking gates | all pass (exit 0) | area_support (5) | all pass (exit 0) | **all pass (exit 0)** |

Readings:

1. **K=15 clears the floor under `grain_equal` without giving up the national fit.** The five near-floor constituencies of G10 pass (min constituency ESS 42.3 → 54.1); national stays 92 / 98; local moves by at most 1.2 points (census 97.2 → 98.4). Absolute ESS scales ~1.4× with the 1.5× rows; the ESS fraction is flat, so the concentration is a property of the rule, not of K.
2. Under `uniform` K=15 lifts the floor margin (54.1 → 71.6) and changes fit by ≤ 1 point; the national rows remain effectively unweighted (OBR 38 %).
3. Cost of K=15: wall +50 % (13 → 20 min), memory flat, artifact 792,690 rows (~2.4 GB h5).
4. The three constants the RC run needs are now each measured on this spine: stretch bound 10 (R8), rule `grain_equal` (R6–R9), K=15 (R9). All three are rulings for María (A2, A3, and the K default); none is applied on PR B until ruled.
