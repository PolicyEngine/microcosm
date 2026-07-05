# PROGRESS — take-up inputs (#312)

Branch: `takeup-inputs` off `origin/main` (7e9a32d).
Sibling: #313 `ecps-parity-gate` (gate/release wiring) — avoid gate-assembly region of release tool; rebase before PR.

## Doctrine (from spec)
- Contract inventory FIRST (blocking): JSON table, engine-version-asserted (same as #301).
- Class (b) seeding: reported-receipt-anchored (#294 pattern) OR calibrated Bernoulli by admin rate.
- HARD RULE: every rate cites an administrative source (FNS/CMS/IRS/ACF, URL+vintage) in stage metadata. Rate-unsourced → leave unseeded, flag in report (stays #313 exemption; follow-up = source as Ledger fact).
- Exclusions: NO SNAP (#294 in flight), NO ACA flag-seeding (model-simulated per #76), NO base-pool regen.
- Ambiguous classification → flag, do not decide (take-up semantics guardian-owned).

## Steps
- [ ] 0. Setup + read #294 pattern, source-stage machinery, #301 metadata guard
- [ ] 1. Determine pinned PE-US version (us extra constraint), install it
- [ ] 2. Contract inventory: enumerate takes_up_* / *_take_up_seed, classify by formula/default
- [ ] 3. Checked-in JSON table + test asserting it vs installed engine
- [ ] 4. Seeding stages for class (b) programs (non-SNAP, non-ACA)
- [ ] 5. Rate provenance in stage metadata; unsourced → unseeded+flagged
- [ ] 6. Diagnostics: per-program participation vs admin count (extend #170)
- [ ] 7. Tests: inventory-vs-engine, per-stage seeding, prove-it-can-find-something
- [ ] 8. Rebase onto origin/main, PR via --body-file (no merge)

## Log
- Setup: worktree created, read spec/DESIGN.md/PR#294 summary. #313 branch not yet pushed to origin.
- Installed pinned PE-US = **1.752.2** (pin `>=1.745.0,<2`).
- CONTRACT INVENTORY (engine-derived, rigorous): 13 `takes_up_*` vars. **ZERO `*_seed` vars. ZERO model-simulated.** ALL 13 have default=True + NO formula = class (b) data-seeded. NONE dead (all consumed via formula or `adds`).
  - BLOCKING FINDING for lead: the `chip_take_up_seed`/`aca_take_up_seed` model-side migration the spec references has NOT landed in the pinned engine. Classifying by installed engine per doctrine.
  - Consumers: snap→snap, tanf→tanf, eitc→eitc(+5 state), dc_ptc→dc_ptc, head_start→head_start, early_head_start→early_head_start, housing_assistance→housing_assistance, medicaid→medicaid_enrolled[adds], chip→chip_enrolled[adds], basic_health_program→bhp_enrolled[adds], medicare→medicare_enrolled[adds], ssi→ssi, aca→person_receives_aca+3.
- SCOPE: exclude SNAP(#294), ACA(spec hard boundary, flag engine mismatch). Seed-candidates gated by PROVENANCE: TANF, EITC, DC PTC, Head Start, Early Head Start, Housing, Medicaid, CHIP, BHP, Medicare, SSI.
- NEXT: per-program admin-rate sourcing (decisive filter); ASEC reported-receipt availability (anchor vs Bernoulli).
- Adapter method `take_up_contract()` + `take_up_variables()` added to policyengine_us.py (+ `_references_variable` helper). Verified vs 1.752.2: all 13 = data_seeded.
- PROVENANCE VERIFICATION COMPLETE (2 parallel agents, primary-source-checked):
  - **TANF 0.22 → SOURCED-ADMINISTRATIVE**: HHS ASPE 24th Report Table 10 Ind.4 = 21.9% (2022). us-data mislabeled vintage 2018. SEED w/ correct vintage.
  - **EITC by #children → SOURCED (IRS NTA/Census)**: NTA 2020 Fig A.7 TY2016: 0=0.65,1=0.86,2=0.85 MATCH; **3+=0.82 NOT 0.85 (us-data fabricated 0.85)**. IRS headline 78% corroborates. SEED w/ CORRECTED 3+=0.82.
  - DC PTC → MODEL-RELATIVE (37,133 real but ÷ PE-model denom; self-referential). rate-unsourced, unseeded.
  - Head Start 40/30 + Early HS 9 → SOURCED-RESEARCH-ONLY (NIEER=Rutgers, NOT ACF/federal); year-split is artifact. Fails admin-source bar → rate-unsourced, unseeded (follow-up: ACF Head Start Program Facts).
  - Medicaid (state) → UNSUPPORTED/MODEL-RELATIVE (0.99=ceiling artifact). CHIP → no standalone current rate. SSI 0.50 → aged-65+ only (49%), scope-mismatch to all-SSI. ALL rate-unsourced, unseeded (#170/#170 follow-up: source as Ledger facts).
  - Medicare → NEAR-UNIVERSAL-BY-DESIGN (~100% correct, seeding not meaningful). Document, unseeded.
  - Housing → no rate ever existed. rate-unsourced, unseeded.
  - ACA → OUT OF SCOPE (spec hard boundary; populace already has native ACA stage in source_stages.json).
- **DECISION: seed exactly TANF + EITC** (both verified administrative/IRS provenance). All others: rate-unsourced OR out-of-scope OR near-universal. This is the honest ledger#77-safe outcome.
