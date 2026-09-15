# UK geography step 2 — atomic-area assignment before/after receipts

Assignment-level measurement of the identity-keyed single-stage atomic-area draw against the legacy sequential ladder draw, at design weights and with no solve, on the canonical spine-r checkpoint. Recorder: `tools/measure_uk_atomic_assignment.py` (diagnostic only; it never gates and release builds do not run it). File name carries the `step2` placeholder until the geography-first umbrella issue is filed; the evidence directory `docs/evidence/uk-geography-step2/` renames with it.

Disclosure: the committed 1 % evidence and this file report population aggregates only. Per-area counts below 3 are written as `<3`, no unit record appears, and the 100 % evidence stays in the licensed spool (its sha256 is recorded here so a reviewer with access can check it).

## R0 — identity

- Branch `uk-geography-step2` (worktree `repos/populace-step2`) on #901's head 2da4f421; commits 3cc384c4 (supports and provenance) and fef75c3f (identity kernel, node swap, CLI, export); this file lands with the harness commit.
- Spine: `spine-r.h5` (791-relationships acceptance checkpoint, 29-stage roster), sha256 `921612e88eefcf21511ac6ad5faf60a1514f577490188c6e39aec49cb4224f27`; 52,846 households (32,754 FRS rows, 20,092 SPI support copies) from 16,288 source households; bound through `full_build_cli.prepare_full_build` with its `.build.json` and `.spine_gates.json` sidecars.
- Ladder: `uk_oa_ladder_2021.npz` sha256 `bed3f13d3a82eea2d1f39248b71c0abf5ba6960a446ddd9415ae1dbcb7ae07fd` (the pinned ladder; rosters and dispersion checks only under the keyed law).
- Supports (`uk/uk_atomic_area_supports.provenance.json`): E&W `1b0dec268be0141f03ec05dc3b55c13733cc2bb20be5e403d76d73c8e27dc797`, Scotland `1649fa831315e6fac80caa4af7f68e837fd985b7f59c2831bb0db8cbfeeee443`, NI `facb9923f589cb68eefa8fad37911e618b6137964ff58a231976ef62584b669f`.
- Assignment definition (seed 42, single-stage law): sha256 `a21816908110cb02dc82aebc373edfef2e85b38c6c7ac70955ac21d343869521`; stream `["sha256-u53-v1", "uk-post-clone-atomic-area-v1", 0, 42]`; identity column `geography_household_key`; identity source `frs`, vintage `2024_25`.
- Chronicle feed `ec7169b` (facts `4a50ee95…`) as pinned; not read by these cells (the subgraph stops at the two geography gates).
- Grid: {legacy, keyed} × {K=1, K=15} × {f001 (sample seed 578, committed), f100 (licensed spool)}; pool seed 42; constituency vintage `2024_pcon`; the same `uk.full.sample` / `uk.full.expand` nodes serve both laws at a given K (cache hits on the second law).
- Both laws run the same UK distribution gate (`uk.full.geography_gate`); the keyed law also runs the shared `uk.full.geography.gate`, whose typed verdict `uk.full.pool` consumes.

## R1 — f001 (1 %, sample seed 578; committed evidence)

Evidence: `docs/evidence/uk-geography-step2/atomic-assignment-cells.json`, sha256 `32f494dff98e81c58fbaa8c03d1a425c1684277e36d4b62757a50f66f44a2a49` (4 cells × 1,011 areas; 985 area counts suppressed as `<3`).

| law | K | rows | source hh | gate | shared gate | London share | const. min rows / ESS / sources | const. max abs z / share abs z>3 (n) | const. breaches | LA min rows / ESS / sources | LA max abs z / share>3 (n) | LA breaches | identity stable | K-growth nested |
|---|---:|---:|---:|---|---|---:|---|---|---:|---|---|---:|---|---|
| legacy | 1 | 502 | 156 | pass | n/a | 0.1303 | 0 / 0.0 / 0 | n/a (0) | 650 | 0 / 0.0 / 0 | 1.96 / 0.0000 (8) | 361 | n/a | n/a |
| keyed | 1 | 502 | 156 | pass | pass | 0.1303 | 0 / 0.0 / 0 | n/a (0) | 650 | 0 / 0.0 / 0 | 1.94 / 0.0000 (8) | 361 | yes | yes |
| legacy | 15 | 7,530 | 156 | pass | n/a | 0.1303 | 2 / 1.7 / 2 | 3.67 / 0.0077 (649) | 650 | 0 / 0.0 / 0 | 2.92 / 0.0000 (355) | 361 | n/a | n/a |
| keyed | 15 | 7,530 | 156 | pass | pass | 0.1303 | 2 / 1.7 / 2 | 4.07 / 0.0031 (649) | 650 | 0 / 0.0 / 0 | 2.77 / 0.0000 (355) | 361 | yes | yes |

Every area breaches the 50 floors at 1 % by construction (the rung exists to exercise the path, not to measure support). The region weight mix is identical across laws at each K, the London share is identical, and the keyed law's production columns equal the in-process re-draw on the same, reversed and clone-0 rows.

## R2 — f100 (100 %; licensed spool)

Evidence (spool, not committed): `f100-cells.json` sha256 `22e909af5f34549817ccd6a62fe334f9b3230e6db9e49b2941817bff448c514d`.

| law | K | rows | source hh | gate | shared gate | London share | const. min rows / ESS / sources | const. max abs z / share abs z>3 (n) | const. max rel err (n≥100) | const. breaches | LA min rows / ESS / sources | LA max abs z / share>3 (n) | LA max rel err (n≥100) | LA breaches | identity stable | K-growth nested |
|---|---:|---:|---:|---|---|---:|---|---|---|---:|---|---|---|---:|---|---|
| legacy | 1 | 52,846 | 16,288 | pass | n/a | 0.1264 | 28 / 12.9 / 27 | 3.38 / 0.0031 (650) | 0.2249 (81) | 145 | 2 / 1.6 / 2 | 3.14 / 0.0028 (360) | 0.2331 (198) | 40 | n/a | n/a |
| keyed | 1 | 52,846 | 16,288 | pass | pass | 0.1264 | 34 / 17.4 / 32 | 3.35 / 0.0015 (650) | 0.2405 (81) | 148 | 0 / 0.0 / 0 | 3.25 / 0.0083 (360) | 0.2401 (197) | 40 | yes | yes |
| legacy | 15 | 792,690 | 16,288 | pass | n/a | 0.1264 | 394 / 339.9 / 354 | 3.14 / 0.0062 (650) | 0.1194 (650) | 0 | 18 / 16.6 / 18 | 2.96 / 0.0000 (361) | 0.0795 (359) | 1 (E06000053) | n/a | n/a |
| keyed | 15 | 792,690 | 16,288 | pass | pass | 0.1264 | 428 / 363.3 / 375 | 3.52 / 0.0046 (650) | 0.1159 (650) | 0 | 17 / 14.1 / 17 | 3.36 / 0.0028 (361) | 0.1068 (359) | 1 (E06000053) | yes | yes |

Reviewed exclusions at design weights, K=15: Isles of Scilly E06000053 breaches all three floors under both laws (legacy 18 rows / ESS 16.6 / 18 sources, keyed 17 / 14.1 / 17, expected 25 under both). City of London E09000001 clears all three floors under both laws (legacy 72 / 55.4 / 72, keyed 75 / 65.4 / 73; expected rows 75.2 legacy, 87.2 keyed): its register entry (ESS 11.0 on the R17 release candidate) is a calibration-time breach, not a design-weight one, so this measurement neither confirms nor retires it.

Cold wall times (first f100 run, before cache; seconds): legacy K=1 locations 0.7, mapping 0.8, UK gate 8.7; keyed K=1 identity 0.5, assign 9.8, derive 0.9, shared gate 1.6, UK gate 8.4; legacy K=15 locations 4.9, mapping 8.7, UK gate 143.8; keyed K=15 identity 13.1, assign 851.6, derive 25.9, shared gate 20.3, UK gate 213.0. Cell walls including population materialisation and measurement: legacy K=15 438 s, keyed K=15 1,391 s. Peak RSS 7.9 GB on the cold run, 10.4 GB on the cached re-run (two K=15 populations materialised). Total for the eight cells: 61 min cold plus 37 min for the re-run that added the shared gate.

## R3 — pre-registered readings

- Constituency-grain expectations identical between laws: **holds** (max absolute difference 0 at K=1 and K=15; the single-stage law and the two-stage law agree exactly at the constituency grain because both draw constituency mass by census households within region).
- LA-grain differences reported per LA: at K=15, 25 of 361 LAs move by more than 2 % and 3 by more than 5 %: E09000001 (City of London) 75.2 → 87.2 (+16.0 %), E07000180 1,379.6 → 1,464.2 (+6.1 %), E07000178 1,487.1 → 1,406.6 (−5.4 %); median absolute shift 0.4 %. The keyed law allocates LA mass by census households instead of by usual-resident population within the drawn constituency, so LAs with atypical persons-per-household move.
- Realized vs expected, max |z| ≤ 4.5 and share |z| > 3 ≤ 1 %: **holds** at every cell (max |z| 3.52, worst share 0.83 %).
- Max relative error ≤ 3 % for expected ≥ 100: **fails as written** at every f100 cell (11.6 %–24 %), under both laws equally. The reading was mis-specified: the maximum over 650 binomial cells of expectation 100–1,500 has a relative sd of 2.6 %–10 %, so a 12 % maximum at K=15 is |z| ≈ 3, which the z readings above already bound. Recorded, not treated as a defect of either law.
- K=15 design-weight breaches: 0 constituency breaches under both laws (**holds**); LA breaches exactly {E06000053} under both laws, not {E06000053, E09000001} (see R2: City of London's exclusion is calibration-time).
- K=1 legacy reproduces #762 R1 (28 / 15.3 / 28; 1 / 1.0 / 1): **not to the row** (28 / 12.9 / 27 and 2 / 1.6 / 2). R1 was measured on a different spine (spine-k, before #791/#903 and the later stage additions); the constituency minimum agrees, ESS and sources move with the spine. No same-spine R1 exists to compare against.
- Keyed within ±10 % of the legacy constituency minima at K=1: 34 vs 28 rows (+21 %), ESS 17.4 vs 12.9, sources 32 vs 27. The band was too tight for a minimum statistic over 650 cells whose expectation is ≈30 rows (sd ≈ 5.5); the shift is +1.1 sd and the two laws' expectations are identical at this grain. At K=15 the minima are 428 vs 394 rows (+8.6 %).
- Region mix and London share identical across laws: **holds** (bitwise-identical weight shares at each K; London 0.1264 at f100).
- Identity stability true (production = in-process re-draw on the same table, on reversed rows, on the clone-0 subset; keys unique): **holds** at every keyed cell.
- K=1 draws ⊂ K=15 draws under the keyed law: **holds** at both rungs.
- Cost ≤ 10 min and ≤ 12 GB at f100/K=15: RSS **holds** (7.9 GB cold, 10.4 GB re-run); wall **fails** for the keyed law (cold geography nodes 1,124 s ≈ 18.7 min, of which `assign_atomic` 852 s and the UK distribution gate 213 s; legacy 158 s). The A7 budget (identity + assign + derive + both gates ≤ 15 min at K=15) is exceeded by 3.7 min: `assign_atomic` iterates households in Python (`iloc` per row); `validate_geography` (20 s) is not the bottleneck. Flagged for Max to vectorise the shared kernel; the UK distribution gate's per-row regex loop (144–213 s, pre-existing) is a UK follow-up.
- Production-path equality (the production columns equal an in-process re-draw): **holds**.

## R4 — disposition

The keyed single-stage law is adopted as the default (`--geography-assignment atomic`) with the legacy draw retained for measurement builds; nothing here certifies a release. Open items carried to the progress log: the `assign_atomic` wall time at K=15 (shared, Max), the City of London exclusion's calibration-time status (unchanged by this PR), the three pre-registration readings that were mis-specified (recorded above so the next PR pre-registers z-based bounds), and the F8 export-surface naming gap.
