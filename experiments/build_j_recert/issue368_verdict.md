## Build J re-certification — SPARSE CERTIFIES with the SSI probe GREEN (+$5.15B); four anti-rot registers digested the week's coverage growth en route

The full gated rebuild is done: base pool rebuild → rmloss100 selection carry-over → sparse release on `main`'s complete stack (#369 coverage gate + reform smoke, #373 SCF asset stage, #384 TANF fix + preflights, #385 release-native loading, SNAP train #350/#352/#353, #387 CMS substitution register) with **every gate on and no bypass flags**. The deployable 57,240-record sparse artifact **passes all certification gates**, and the reform-coverage smoke — the gate this issue exists for — scores the SSI $10k/$20k reform at **+$5.15B, nonzero and green**. An SSI resource-limit reform no longer silently scores $0. STAGING/LOCAL ONLY throughout; **publication and the default flip are Max's call** — nothing touched prod HF or `latest.json`.

Certified run: `populace-us-2024-buildj-sparse-rmloss100-75d5add-20260710T094201Z` (rc=0, `release_gates.passed=True`, `failures=[]`), branch `build-j-recert`, policyengine-us **1.764.6** / core **3.26.11**, seed 0, period 2024. All numbers below verbatim from the run's `calibration_diagnostics.json` / `input_mass_parity.json` / `reform_coverage_smoke.json` / `build_manifest.json`.

### The SSI probe (Deliverable 1's smoke, on Deliverable 2's data)

| probe | result |
|---|---|
| `ssi_asset_limit_10k_20k` ($10k individual / $20k couple) | **effect +$5,153,370,118 @ 2024** — baseline SSI $55,192,291,256 → reform $60,345,661,375 — **PASSED** ($1B floor, enforced) |

Against the #374 expectations: the re-solved rebuild lands **below** the +$9.66B SCF-only overlay (the predicted re-solve direction) and **above** the ~+$1.60B dense-native reference class — the gap is the **SIPP blend, the known open refinement** (#374 step 1: SCF-only understates the low-liquid-asset left tail, so restored baseline recipients run thin and the reform delta runs high). The #369 acceptance criterion (smoke green on a bound reform) is met; landing the magnitude in the dense-native class is #374's remaining work, not this certification's.

The three SSI countable-resource columns (`bank_account_assets`, `stock_assets`, `bond_assets`) are on the exported H5 with signal — they are **hard requirements** in the coverage manifest (no exclusion), exactly as this issue demanded.

### Headline diagnostics vs Build I (bar: 0.030833 / 0.8888 / +0.22% / 0-of-35)

| metric | **Build J sparse** | Build I (rmloss100) | Build H frozen |
|---|---:|---:|---:|
| final loss | **0.030843** | 0.030833 | 0.030915 |
| within-10% | **0.8900** | 0.8888 | 0.8901 |
| ESS | 12,088.6 | 12,303.4 | 13,184 |
| realized max ratio | 4.994 | 4.993 | — |
| records / nonzero | 57,240 / 57,240 | 57,240 / 57,240 | 57,240 |
| fed income tax (SOI ht2 liability) | +0.58% | +0.22% | +0.45% |
| SS benefits (SSA) | −0.22% | −0.18% | −0.18% |
| net capital gain (CBO) | −25.69% | −24.18% | −26.65% |
| mortgage tax-exp (JCT) | +38.59% | +36.73% | +35.48% |
| **export-mass parity** | **PASS 0 / 35** | PASS 0 / 35 | FAIL 1 / 35 |
| zero-support (post-exclusion) | **0** | 0 | 0 |

Export-mass detail (±50% band, $1B floor; reviewed exclusions `estate_income` + `non_sch_d_capital_gains` only, both used, none unused): `miscellaneous_income` **+11.46%** (Build I +11.5% — the Build I fix holds), `home_mortgage_interest` **+28.92%**, `first_home_mortgage_interest` **+29.00%** — all in band. The export's candidate-only set now carries the restored surface: the three asset leaves, `health_savings_account_ald`, `spm_unit_pre_subsidy_childcare_expenses`, the SNAP-train inputs (hours, pregnancy, ABAWD exemptions), and the seeded take-up flags.

Comparison caveat, stated: the registry is **not** byte-identical to Build I's — Build J compiles **5,510** specs (`091ace02f962`) vs Build I's 5,514 (`d71c59514e3a`). Delta = the **#387 RI CMS substitution** (+1 injected spec: RI anchored at the ledger-verified November 2024 count 273,400, issue-linked to #386, cannot-rot on backfill — recorded in `build_manifest.json`) and **5 new TANF thin-state zero-support exclusions** (−5; below). Base pool data is identical to Build F/H/I (`base_household_weight_total` exact match; base h5 sha `0b50660a` differs from `18833fb6` only by HDF5 serialization non-determinism); selection is exactly Build I's rmloss100 57,240 (identity join: 0 unmapped, 0 ambiguous).

### Gate table (all enforced, no bypass flags)

| gate | result |
|---|---|
| structural: target_profile, health_input, base_population, immigration, hours_worked, snap_take_up, eligibility_inputs, pregnancy, snap_discretionary_exemption | **PASS** (all) |
| #384 register-consistency + take-up preflights | **PASS** |
| #334 Medicaid take-up (with #387 RI substitution applied, not stale) | **PASS** |
| #337 eCPS parity (register 93 → 88 after the five stale-exemption removals) | **PASS** |
| zero-support (24-cell reviewed list: Build I's 19 + 5 TANF; post-exclusion 0) | **PASS** |
| #369 input coverage (70 required incl. the 3 asset leaves; degenerate-required none) | **PASS** |
| degenerate input (#286; TANF unexcused and non-degenerate) | **PASS** |
| export-mass parity (35 columns) | **PASS 0/35** |
| #369 reform-coverage smoke (SSI probe) | **PASS +$5.15B** |
| take-up contract | **PASS** |

### The four-register night (one root cause, four catches — the anti-rot machinery digesting the week's coverage gains)

Build J is the first release-scale run of most of this week's merges, and four registers failed closed in sequence; every failure was diagnosed against the actual frame/artifact before the register moved:

1. **#334 Medicaid take-up**: RI (FIPS 44) had no CMS enrollment target — a genuine source hole (CMS PI dataset carries RI 202412 = 0, footnoted "Unable to Provide Data due to System Limitations"; RI never reported the month). Filed **#386**; fixed by **#387**'s reviewed substitution register (nearest reported month, 2024-11 = 273,400, cannot-rot on backfill). The gate refused to ship RI silently anchored-only — correct behavior, first time out.
2. **#337 eCPS parity staleness**: five exemptions rotted because the candidate now populates the layers — verified per-layer on the rmloss100 57k with the gate's own share computation: `block_geoid`/`tract_geoid`/`county_fips` at **1.0 = reference 1.0** (the #277/#289 geography-ladder spine), `spm_unit_pre_subsidy_childcare_expenses` 0.0755 vs ref 0.1022, `health_savings_account_ald` 0.00936 vs ref 0.0962 (genuine but thin; depth stays tracked in #32). Register 93 → 88.
3. **Zero-support**: 5 TANF state cells (`fl/ia/mo/ne/nj`) lost support because **#384 made TANF take-up seeding live** (13,205/59,900 spm_units True vs the old constant-True) — verified at 0 nonzero records on the identity-matched staged frame while every other state carries 1–14 (national 156). The Build I AL precedent, five states wider under real take-up; the dense parent expresses these cells; selection-side remedy is #346/#355-class work. Exclusion list 19 → 24, each with the measured justification.
4. **#369 input-coverage staleness**: the coverage manifest (which derives from the parity register) still excluded the same five columns — regenerated with `tools/build_us_release_input_coverage_manifest.py`; diff = exactly five `reviewed_exclusion` → `required` promotions (65→70 required / 93→88 exclusions). A one-pass sweep of every remaining register (degenerate, documented-absent, take-up contract, substitutions, probes) found no other entry carrying the five columns or TANF-era assumptions.

### Verdict: the sparse artifact is **CERTIFIABLE**

All gates pass with everything enforced; final loss and within-10% land at the Build I bar (within-10% beats it); export-mass parity holds 0/35 with the same two reviewed exclusions; the SSI probe — the reason this issue exists — is green at +$5.15B with the asset columns as hard requirements. The dense diagnostic arm (full 337,704 pool, same stages, no zero-support exclusions) is running serialized — its numbers follow in this thread when the chunked solve completes.

Remaining follow-ups, tracked: #374 (SIPP blend + landing the SSI magnitude in the dense-native class), #386/#387 (RI substitution retires automatically if CMS backfills), #32 (HSA/childcare depth), #346/#355-class (thin-state TANF carriers in selection). The register edits made during this run (parity −5, coverage-manifest promotion, 24-cell zero-support list) live on `build-j-recert` and need a main PR alongside any publication decision.

Cache-key: base `0b50660a21e0…` (pool data == Build F `18833fb6`); pe-us 1.764.6 / core 3.26.11; seed 0; period 2024; feed v8 `94b7155f`; registry `091ace02f962` (5,510 compiled); selection `152baca3` (57,240; 0/0 join); zero-support exclusions `0be0c200…` (24 cells); export-mass ref `c2065b64`; SCF donor member `6b8dd2d9` (zip `3bb4d890`, both == #373 pins). Release id `populace-us-2024-buildj-sparse-rmloss100-75d5add-20260710T094201Z`: `release_manifest.json` sha256 `3d292c2231d918cf58b68d14bfabd0a1ea252a43ae1890cec5228bd10db21ba9`, exported H5 sha256 `9d1e460ddbc49c9b467b936f07edd4576ae2667ef39fdea2490bc5577c038d51`. Artifacts under `~/PolicyEngine/_buildj-runtime/out/buildj-run/sparse/`; working log `PROGRESS_BUILDJ.md` on `build-j-recert`.
