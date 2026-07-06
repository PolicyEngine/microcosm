# Build G certification candidates — assembly + run — PROGRESS

Task: PolicyEngine/populace **#299 Build G run entry**. Assemble and run the two Build G
certification candidates on the now-complete prerequisite stack (#317/#321/#324/#327/#328 + PR #330):
- **SPARSE headline** = frozen-57k-support selection (committed manifest) + dense polish at production
  defaults (λ=0) — the ~57k laptop artifact.
- **DENSE parent** = full 337,704 base, mass gate against the corrected (live-default 57k) reference.

**STAGING/LOCAL ONLY.** Never touch policyengine/populace-us prod repo, `latest.json`, or prod HF write
paths. Publication is Max's call. All long compute DETACHED (nohup + pidfile + logs).

Worktree: `/Users/maxghenis/PolicyEngine/_worktrees/populace-build-g-run` (branch `build-g-run`,
**fresh off origin/main HEAD 2bf603b = the #330 merge commit** — NOT the opener's stale `build-g`).
Runtime home (OUTSIDE repo): `/Users/maxghenis/PolicyEngine/_buildg-runtime/` (durable; reused from opener + #330).

## Verified prerequisites (read before any change)
- **#330 MERGED** to main (merge commit 2bf603b, 2026-07-06T22:42Z). `warm_start_selection.py` +
  `build_us_selection_source_manifest.py` + release-tool selection wiring present at origin/main.
- **Base F H5** (REUSE, sha VERIFIED): `_worktrees/populace-build-f/out/base-f-20260705/base_populace_us_2024_puf_support.h5`
  = `18833fb68e60ee74461608d81a5c5ab7d52435e17026d9e3b062d9de18d6871f` (matches required 18833fb6).
  3-year ASEC pool (2024+2023+2022 equal thirds) → 2× PUF clone = 337,704 hh. All 4 #278 leaves present.
- **Committed selection manifest** (from #330): `_buildg-runtime/inputs/certified_57k_selection_source.json`
  — 57,240 identities, `identities_sha256=77363a47…`, join_key = `[source_year, source_household_id,
  household_support_channel, household_support_clone_index]`, source sha `c2065b64…`, HF repo/revision
  `policyengine/populace-us @ …sparse-l0-refit-57k-…-national-only-20260701`.
- **Live-default 57k H5** (the #327 reference, ON DISK): `_buildg-runtime/forensics/populace_us_2024.h5`
  sha `c2065b64…` (354 MB) — the exact HF snapshot the selection manifest points at.
- **v7 feed** (REUSE, sha VERIFIED): `_buildf-runtime/inputs/consumer_facts_buildf_v7.jsonl` = `735f326a…`
  (Build F's feed: v5 curated surface + aging − M-CHIP CHIP rows at all periods). 63 CHIP rows = 32 S-CHIP
  states only (0 M-CHIP), periods 2024-12 + 2025-12.
- **#330 validation result (sparse prior)**: frozen-support + dense polish on base-F reproduced
  certified-grade calibration — loss **0.0315** (beats certified 0.044), within-10% 88.1%, income tax
  −4.1%, SS −0.1%, mortgage tax-exp +43.1%, ESS 13,423; rc=1 on **39 zero-support targets**
  (20 M-CHIP CHIP + 18 SOI under_1.taxable_interest + 1 AL TANF). Certified reference bands:
  sparse 0.044 / 94.7%; dense 0.0423 / 86.4%.

## Certified reference bands (the verdict yardsticks)
- SPARSE headline: final_loss ~0.044, within-10% ~94.7% (live-default 57k). #330 seam prior: 0.0315 / 88.1%.
- DENSE parent: final_loss 0.0423, within-10% 86.4% (f0af251). Build F dense reached 0.0416 / 86.1%.

================================================================================
## STEP 1 — thread #327's one-line fix (reference-frame into `_export_input_mass_gate`)
================================================================================

**Code fact (origin/main, tools/build_us_fiscal_refresh_release.py):**
- `_export_input_mass_gate(export_frame, base_frame, *, relative_tolerance, minimum_reference_total)`
  (line 3527) compares `us_input_mass_totals(export_frame)` vs `us_input_mass_totals(base_frame)` — the
  RAW pre-calibration base. Called at line 5553 with `base_frame`.
- `--input-mass-reference-h5` (line 567) currently feeds ONLY `_input_mass_reference_gate` (base-vs-ref),
  NOT the export gate — exactly the attempt-6 finding.
- The underlying `input_mass_parity_gate` (gates.py:978) ALREADY accepts `reviewed_exclusions` and a
  reference-totals arg. `load_us_frame` (l0_refit_export.py:286) loads a Frame from an H5.

**Change (per #327 reference-decision comment — use the LIVE DEFAULT 57k, not f0af251):**
1. Give `_export_input_mass_gate` a new keyword `reference_frame: Frame | None = None`; when provided,
   compute reference totals from it; when None, fall back to `base_frame` (PRESERVES current behavior
   exactly — no reference H5 ⇒ raw-base comparison as today).
2. At the call site, pass `reference_frame = _load_export_mass_reference(args.input_mass_reference_h5)`
   (a cached `load_us_frame` of the reference H5), falling back to base_frame when the flag is absent.
   The reference label in messages becomes the reference H5 name (or "base_frame" when absent).
3. Wire the SPARSE + DENSE runs with `--input-mass-reference-h5 <live-default 57k>`
   (`_buildg-runtime/forensics/populace_us_2024.h5`). Per #327: this vindicates 11/14 mis-referenced
   columns cleanly. The 3 residuals (mortgage ×2, miscellaneous_income) are the cold-vs-informed gap —
   which the FROZEN-SUPPORT construction (this build's sparse) resolves (mortgage tighter than cold dense;
   #330 showed +43.1% vs +59.7%). **VERIFY on the actual run — do not assume; if any residual still
   fails legitimately, REPORT rather than exclude.**

**Unit test (add to test_us_fiscal_refresh_builder.py):** gate PASSES when export≈reference despite
export≫raw-base (the #327 case); gate still FAILS on genuine drift vs reference (a zeroed/halved column).

================================================================================
## STEP 2 — feed reconciliation for the 39 zero-support targets — DESIGN PROPOSAL
================================================================================

**The 39 targets (from #330 validation diagnostics, exact):**
- **20 M-CHIP CHIP** — `cms_medicaid.month2024_12.state_enrollment.{ak,ca,dc,hi,il,ky,md,me,mi,mn,nc,nd,
  ne,nh,nm,oh,ok,sc,vt,wy}.total_chip_enrollment@2024`. Class #170/#321: M-CHIP states fold CHIP into
  Medicaid; PE-US materializes SEPARATE-CHIP only ⇒ structurally zero support **for ANY artifact**
  (dense and sparse alike).
- **18 SOI** — `irs_soi.ty2022.historic_table_2.state_agi.{al,ct,in,me,ms,ne,nm,ri,wy}.under_1.
  taxable_interest_{amount,returns}@2024` (9 states × 2). Narrow under-$1-AGI offset-income tail cells.
  Same class as the ia/nd cells ALREADY in `US_FISCAL_TARGET_SUPPORT_EXCLUSIONS`.
- **1 TANF** — `hhs_acf_tanf.fy2024.cash_assistance.al.basic_assistance_…all_funds@2024`. Same class as
  the 22 TANF states ALREADY excluded (ar/az/co/…). AL has zero positive TANF support on this base.

**M-CHIP DIAGNOSIS (required by task — "if they reappear on the 57k support, diagnose"):**
The v7 feed is CORRECT — it carries 0 M-CHIP CHIP rows (only 32 S-CHIP states). BUT the #321 **code-side
derivation-skip** fix (`_M_CHIP_STATE_FIPS` skip in `_with_derived_chip_enrollment_targets`) landed only on
the `build-f` branch (commits 03cc3d9, 9e25d54) and was **NEVER MERGED to main**. On main,
`_with_derived_chip_enrollment_targets` (fiscal_targets.py:818) re-derives CHIP = (combined − Medicaid) for
any (period, state_fips) that has both control rows and no direct CHIP row — which recreates the 20 M-CHIP
targets. So the 20 reappear because the derivation fallback is unskipped on main. (Note: Build F attempt-3
observed the derivation NOT firing because the feed's medicaid control role was `total_medicaid_enrollment`
not `medicaid_enrollment`; whether it fires here depends on the v7 control-row roles — VERIFY on the run.)

**DESIGN DECISION — per-artifact target applicability vs a single pruned v8 feed. RECOMMENDATION: per-artifact.**

Rationale:
- The current `US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` (fiscal_targets.py:433, 42 entries) is **global /
  per-feed**: a `source_record_id` in the dict makes `_reference_from_ledger_fact` return None, dropping
  the target from EVERY artifact built from that feed. That is the "single pruned v8 feed for both" model.
- But the 18 SOI + 1 TANF cells are zero-support **because the 57,240 sparse support is too thin to
  populate them** — the 337,704 DENSE parent has 6× the records (incl. the state tails) and (per Build F
  attempt-5) passed the degenerate/zero-support gate on everything EXCEPT the 20 M-CHIP. Excluding the SOI
  /TANF cells globally would wrongly prune from the dense parent targets it can actually hit. The task
  brief explicitly biases toward per-artifact ("the dense parent may legitimately support targets the
  sparse can't").
- The 20 M-CHIP are structural (#170) and un-expressible by BOTH ⇒ they belong in the GLOBAL registry.

**Therefore the recommended construction is a hybrid, all reason'd + ledgered as a supersession:**
1. **Global (both artifacts):** add the 20 M-CHIP `total_chip_enrollment@2024` source_record_ids to
   `US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` with the #170/#321 reason. This is the correct home — they are
   un-expressible regardless of support density, and it makes main consistent with the build-f #321 intent
   without needing the derivation-skip code (excluding the source_record_id short-circuits before the
   derivation for direct rows; the derived rows require a separate guard — see implementation note).
2. **Per-artifact (sparse only):** a declared, reason'd **sparse exclusion set** for the 18 SOI + 1 TANF
   support-expressibility cells, applied ONLY to the sparse/57k artifact, NOT the dense parent. Mechanism:
   a `--zero-support-exclusions <json>` CLI arg on the release tool that augments the global registry for
   THIS run only, recorded in the manifest as a `zero_support_exclusions` provenance block (source, reason,
   derivation = "support-expressibility on the 57,240 frozen support").
3. **Verify-don't-assume:** run the DENSE parent FIRST (or in parallel) and read its zero-support list. If
   the dense parent ALSO shows the 18 SOI + 1 TANF as zero-support, they are NOT dense-expressible either
   and should move to the global registry (single-feed prune becomes correct for those too). If the dense
   parent expresses them (nonzero support), the per-artifact split is vindicated. **The empirical dense
   zero-support list decides the SOI/TANF home; document either way as a ledgered supersession.**

Implementation note on the M-CHIP derived rows: the global `US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` check in
`_reference_from_ledger_fact` fires on DIRECT fact rows only. If the 20 M-CHIP appear via the DERIVATION
(combined − Medicaid), excluding the direct source_record_id will not stop them. The clean, main-consistent
fix is to port the #321 derivation-skip guard (skip M-CHIP state FIPS in `_with_derived_chip_enrollment_
targets`) — this is the RIGHT change (it is exactly what build-f did) and closes the loop on #321. Decide
after confirming empirically whether the 20 arise from direct rows (feed) or the derivation (code) on THIS
run. **PROPOSE first; port the guard only if the derivation is the confirmed source.**

================================================================================
## STEP 3 — PIN policyengine-us 1.764.x (per #324 GO)
================================================================================
- Incumbent (Build F): 1.752.2. Spec `>=1.745.0,<2`. 1.764.x available: 1.764.0…1.764.6. 1.765.x exists
  but the opener flagged 1.765.x adds PA/NJ/CO/CA/AZ state-tax fixes needing a separate re-run ⇒ stay in
  the GO's named 1.764.x window.
- Pin to the LATEST 1.764.x = **1.764.6** (contains 1.764.1's NY IT-196 §68 wiring = the NY re-seat the
  opener attributes; the opener's battery ran at 1.764.0 but confirmed the NY fix on-disk in the 1.764 line).
- Env: `uv sync --all-packages --extra us` after `uv lock --upgrade-package policyengine-us` to move the pin.
  RECORD the exact resolved version. Expect NY state income tax +~28.5% and national state-tax +~4.8% re-seat
  purely from the model (1.764.1). **Attribute per the opener's #324 table; VERIFY the realized NY number
  from the run's diagnostics.**

================================================================================
## STEP 4 — RUN BOTH CANDIDATES DETACHED (chain script; nohup + pidfile + logs)
================================================================================
- SPARSE = `--selection-source-manifest <committed 57k manifest>` + `--dense-default-dataset` (frozen-support
  + dense polish = the certified construction per #330) + `--input-mass-reference-h5 <live-default 57k>` +
  per-artifact sparse zero-support exclusions. Checkpoint root `_buildg-runtime/checkpoints/buildg-sparse`.
- DENSE = full base (no selection) + `--dense-default-dataset` + `--input-mass-reference-h5 <live-default 57k>`.
  Checkpoint root `_buildg-runtime/checkpoints/buildg-dense`.
- BOTH: `--base-h5 <base-f 18833fb6>`, `--ledger-facts <v7>`, `--seed 0`, `--no-staging`, `--out
  _buildg-runtime/out/buildg-run`. Checkpoint roots OUTSIDE repo. NEVER touch prod HF / latest.json.
- NOTE: pin bump 1.752.2→1.764.6 invalidates the 1.752-era checkpoints (version is in the cache key) —
  fresh checkpoint roots, compile re-runs at the new pin. Correct + expected.

================================================================================
## STEP 5 — GATE TABLES (per artifact) + STEP 6 — REPORT + #299/#324
================================================================================
Full pass/fail table per artifact vs certified bands (sparse 0.044/94.7%; dense 0.0423/86.4%); realized
max-weight-ratio; all cache-key components; #330 numbers as the sparse prior. CERTIFIABLE-or-not verdict
per artifact. Post to #299 (Build G run entry) + update #324. Publication is Max's.

## HARD RULES
- **Three-strike stop**: diagnose precisely and REPORT rather than iterate past 3 failed attempts.
- Detached compute only; commit + PROGRESS every step; if any gate failure is NEW, REPORT before changing.

================================================================================
## TIMELINE (append every step)
================================================================================

### 2026-07-06 — setup + design complete
- Read full prereq stack: #299 (Build E/F campaign, both closing comments), #324 + GO, #327 + reference
  decision, #328 + PR #330, #321, Build F PROGRESS, opener PROGRESS_BUILDG, #330 agent PROGRESS_BUILDG_328,
  relaunch harness, build_f_chain.sh, committed selection manifest.
- Verified: #330 in origin/main; base-F sha 18833fb6; manifest 57,240 identities; live-default 57k on disk;
  v7 feed 735f326a (0 M-CHIP rows). Diagnosed 39 zero-support = 20 M-CHIP (derivation, #321 code-fix not on
  main) + 18 SOI tail + 1 AL TANF (support-expressibility). pe-us 1.764.x = 1.764.0…1.764.6 available.
- Fresh worktree `build-g-run` off origin/main 2bf603b. Runtime subdirs created.
- DESIGN: step-1 reference-frame fix (default base, wire live-default 57k); step-2 per-artifact applicability
  (global M-CHIP + sparse-only SOI/TANF), verify against dense zero-support empirically.
- NEXT: implement step 1 (code + test), decide step 2 after dense empirics, pin 1.764.6, run both detached.
