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

### 2026-07-06 — STEP 1 + STEP 2 implemented; step-2 design RESOLVED empirically
**Env**: `uv lock --upgrade-package policyengine-us==1.764.6` + `uv sync --all-packages --extra us`
resolved **pe-us 1.764.6 / core 3.26.11** (exact pin recorded per #324 GO; 1.764.6 = latest 1.764.x,
includes 1.764.1 NY IT-196 §68). Env marker written.

**STEP 1 committed (72d80ca)**: `_export_input_mass_gate` now takes `reference_frame` (defaults to
raw base — behaviour preserved), `reference_name`, `reviewed_exclusions`; call site wires it from
`--input-mass-reference-h5`. 3 unit tests pass (default raw-base fail; passes vs certified ref despite
export>>base; still fails genuine #278 zeroing/drift vs ref).

**STEP 2 M-CHIP (global) — cherry-picked #321 fix (bfb234f, 5265757)**: DIAGNOSIS CONFIRMED — the v7
feed carries 0 direct M-CHIP CHIP rows (correct), but on main `_with_derived_chip_enrollment_targets`
re-derives CHIP=(combined−medicaid) for M-CHIP states (v7 has BOTH control rows for all 20). The #321
derivation-skip (`_M_CHIP_STATE_FIPS`, 20 FIPS = exactly the 20 M-CHIP zero-support states) was never
merged to main — it lived only on build-f. Ported both #321 commits (skip + tests) cleanly. 10 CHIP/
zero-support/reviewed tests pass (incl. the 42-entry global-registry invariant, untouched).

**STEP 2 SOI/TANF (per-artifact) — DECISION RESOLVED, empirically grounded.** Mechanism added:
`--zero-support-exclusions <json>` on the release tool → `extra_support_exclusions` threaded through
`compile_us_fiscal_target_registry` → `_dynamic_us_fiscal_target_references` → `_reference_from_ledger_fact`;
recorded in `us_source_coverage.json` as `fiscal_target_support_exclusions_per_run` (never mutates the
module constant). **DECISIVE EVIDENCE for per-artifact (not single pruned feed):** Build F attempt-5
DENSE (337k base, SAME v7 feed, WITH the M-CHIP skip) had **0 zero-support targets** (release_gates
passed=True; 0 SOI, 0 TANF, 0 CHIP). So the 337k dense parent EXPRESSES the 18 SOI under_1.taxable_interest
+ 1 AL TANF cells; only the 57k sparse frozen support cannot. This is exactly the "dense may legitimately
support targets the sparse can't" case. **Ledgered supersession:**
  - Dense parent → NO per-run exclusions (only the global M-CHIP derivation skip). Full v7 registry.
  - Sparse → the 19-cell exclusion file (18 SOI + 1 AL TANF), reason = support-expressibility on the
    57,240 frozen support. Same class as the ia/nd SOI + 22 TANF states ALREADY in the global registry —
    but kept per-run because they ARE dense-expressible (unlike the global ones on the national support).
  - The 20 M-CHIP are GLOBAL (structural #170; un-expressible on ANY support) via the derivation skip.

n_targets note: #330 sparse validation showed 5541 targets (on 328-validate branch = NO M-CHIP skip →
re-derived 20 M-CHIP); Build F dense attempt-5 = 5521 (M-CHIP skipped). With the skip now on build-g-run,
sparse drops the 20 M-CHIP too → remaining sparse zero-support = 18 SOI + 1 TANF = the 19 in the exclusion
file. Dense = 0 remaining.

**Full populace-build suite launched detached** (build_suite.pid) to confirm no regressions before compute.

### 2026-07-06 ~23:10 ET — RUN 1 (dense) failed early on the BASE-vs-reference gate — DIAGNOSED + FIXED (scope error in my own step-1 wiring)
**Symptom:** dense run rc=1 right after ACA materialization (before calibration), on
`_input_mass_reference_gate`: `base_frame mass 2.12e11 vs populace_us_2024.h5 8.64e11 (-75.5%)` for
LT cap gains + 7 more PUF columns (estate_income -68%, non_sch_d -55%, partnership -80%, qual div -55%,
rental -51%, ST cap gains -75%, misc -117%).

**Diagnosis (precise):** `--input-mass-reference-h5` arms BOTH gates —
(1) `_input_mass_reference_gate(base_frame, reference)` = the #278 base-LOSS guard, comparing the RAW
    PRE-calibration base to the reference; and (2) my #327 export gate. The live-default 57k reference is
    CALIBRATED (its PUF income is scaled to SOI/CBO). The raw 337k base structurally UNDER-reports those
    columns (CPS under-report + PUF seeds low), so base << calibrated-reference by -50% to -117% on exactly
    the columns calibration is meant to fix → the base-vs-reference gate over-fires. This is precisely the
    Build F attempt-6 pre-flight finding #1-2 ("the flag adds a new base-vs-reference gate that itself trips").
**Root cause:** MY step-1 wiring reused `--input-mass-reference-h5` for the export gate, which also feeds
    the base-vs-reference gate. #327's verdict was EXPORT-gate-only; it never intended to arm the
    base-vs-reference gate with a calibrated reference.
**Fix (scope-correct, minimal):** added a DEDICATED `--export-input-mass-reference-h5` flag for the export
    gate; `--input-mass-reference-h5` stays for the base-vs-reference gate (left OFF/None for these runs,
    since a raw base legitimately undershoots a calibrated reference). Chain updated to use the new flag.
    export-gate unit tests still pass; --help shows both flags. NOT a semantics change to any gate — a
    scope correction of my own wiring. Caught in the first ~3 min (before any calibration burned).
**NOT a three-strike build failure:** this was a harness misconfiguration, corrected before calibration.
    Relaunching with the corrected flag. Base-vs-reference gate is intentionally not armed (documented).

### 2026-07-06 ~23:14 ET — RUN 2 launched (corrected flags); dense-export outcome PRE-VALIDATED
Relaunched (commit 38457f8): dense first, then sparse; --export-input-mass-reference-h5=live-default 57k;
base-vs-reference gate NOT armed. Dense healthy (99% CPU), past the run-1 failure point.

**Pre-validation of the dense export gate (using Build F attempt-5 dense export masses vs the measured
live-default 57k reference, tol +-50%):** 11/14 columns in-band, exactly matching #327's reference-decision
table; 3 residuals FAIL = first_home_mortgage_interest (+62.5%), home_mortgage_interest (+70.7%),
miscellaneous_income (-62.9%). These are the documented #328 cold-calibration residuals (mortgage
tax-expenditure overshoot + sign-unstable misc), NOT gate mis-references. Per #327: "leave the
mortgage/misc residuals to #328 rather than granting blanket exclusions." So the DENSE parent is
predicted NOT certifiable as-is (3 genuine cold-calibration export-mass residuals) — but every structural
gate + zero-support (0 with the full v7 registry) passes. Exact numbers will shift at 1.764.6; pattern holds.

**Sparse prediction:** the frozen-support construction fits the mortgage tax-expenditure tighter (#330:
+43.1% vs cold dense +59.7%), so the sparse export-mass residual should be smaller — possibly clearing the
mortgage columns. The run will tell. Sparse zero-support after the M-CHIP skip + 19-cell exclusion = 0.

RUN 2 monitored (task armed). Both diagnostics write before the export gate raises, so gate tables are
extractable even if the export gate fails on residuals.

### 2026-07-06 ~23:18 ET — RUN 2 both artifacts died on the STALE DEGENERATE REGISTER (known prereq, not novel)
Both dense AND sparse rc=1 on `_degenerate_input_signal_gate` (#286), BEFORE calibration (base-frame gate).
Two documented items, exactly the #330/#324-flagged register refresh never merged from build-f:
1. `second_home_mortgage_{balance,interest,origination_year}` = constant engine default → need reviewed
   exclusion (un-imputed trio → populace#38).
2. Stale reviewed exclusions `['takes_up_eitc','takes_up_tanf_if_eligible']` — carry #315-seeded signal
   now → must be RETIRED.
**Positive signal:** the dense run PASSED the base-vs-reference gate (advanced to the degenerate gate,
which runs after it) — confirming the RUN-2 two-flag scope fix WORKED. Both runs failed fast (~2 min),
no wasted compute.
**Fix (pre-approved, exact build-f b5980d2 edit):** ported the register edit to
`US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS` — removed the two stale take-up flags, added the second_home
trio (→ #38). 8 degenerate/take-up/export tests pass. This is the register refresh #330 applied only on a
throwaway branch; now on build-g-run. Relaunching.

### 2026-07-06 ~23:19 ET — RUN 3 launched (ALL prerequisites fixed); genuine calibration attempt
Verified via `git log origin/main..build-f` that ALL build-f base-gate prerequisites are now ported to
build-g-run: #321 M-CHIP derivation skip (cherry-picked), register refresh b5980d2 (committed). No other
un-ported base-gate config exists. So run 3 is the FIRST genuine calibration attempt — base gates
(base-vs-ref [not armed], degenerate [register fixed], eCPS, take-up) all pass; the 3 earlier rc=1s were
sequential PREREQUISITE/harness fixes (each caught in ~2-3 min before any calibration), not strikes against
a genuine certification issue.

commit 129165b, pe-us 1.764.6, all shas verified. Dense first (full 337k → base gates → target
compilation → 1500-epoch calibration → export gate vs live-default 57k reference). Then sparse
(frozen-57k manifest + dense polish + 19-cell exclusions). Monitor armed (b3mo82gz0). Dense healthy
(100% CPU, ~3 min in ACA materialization as of launch+3min).

PREDICTED outcomes (to verify against the run):
- DENSE: all structural gates PASS; 0 zero-support (full v7 registry, M-CHIP skipped); export-mass gate
  vs live-default 57k → 11/14 in-band, ~3 residuals (mortgage×2 +~65%, misc −63%) = the #328 cold-calib
  overshoot. Loss ~0.042, within-10% ~86% (matching Build F dense 0.0416/86.1%, shifting slightly at 1.764.6).
  → NOT certifiable as-is (3 genuine export-mass residuals) unless the residuals resolve at the new pin.
- SPARSE (headline): 0 zero-support (M-CHIP skip + 19-cell exclusion); frozen-support construction fits
  mortgage tighter (#330: +43.1% vs cold +59.7%) so export residuals should be SMALLER — possibly clearing
  mortgage. Loss ~0.031 (#330 prior 0.0315), within-10% ~88%; income tax ~−4%, SS ~0%. → likely the
  certifiable candidate if export-mass clears.

### 2026-07-07 ~01:10 ET — RUN 3 RESULTS (both artifacts calibrated; export-mass residuals vs CORRECTED reference)
Both dense + sparse reached the export gate (calibration completed, all base+structural gates PASS,
0 zero-support) and rc=1 ONLY on the export input-mass gate — now correctly vs `populace_us_2024.h5`
(the live-default 57k), confirming the #327 fix works end-to-end.

**DENSE parent (populace-us-2024-buildg-dense-129165b):**
- final_loss **0.04139** (BEATS f0af251 0.0423), within-10% 86.16%, ESS 81,206, realized max-ratio 5.0,
  n_records 337,704. ALL structural gates PASS; 0 zero-support (M-CHIP skip + full v7 registry, 5521 targets).
- export-mass vs live-default 57k: **4 fails** — first_home_mortgage_interest +61.8%, home_mortgage_interest
  +69.2%, miscellaneous_income −68.3% (the predicted 3 #328 residuals) + **estate_income −53.6%** (NEW).

**SPARSE headline (frozen-57k, populace-us-2024-buildg-sparse-129165b):**
- final_loss **0.02964** (BEATS certified 0.044 AND the #330 prior 0.0315), within-10% 89.15%, ESS 13,388,
  realized max-ratio 5.0, n_records 57,240 (seam recovered exactly). ALL structural gates PASS; 0 zero-support
  (M-CHIP skip + 19-cell exclusion, 5502 targets). Selection provenance fully recorded (frozen_support mode).
- export-mass vs live-default 57k: **4 fails** — estate_income −66.8%, home_mortgage_interest **+52.4%**
  (TIGHTER than dense +69.2%; first_home dropped OUT — confirms #330's frozen-fits-mortgage-tighter
  prediction), miscellaneous_income −54.3%, non_sch_d_capital_gains +111.4%.

**estate_income + non_sch_d analysis (raw base / reference masses):** estate_income raw base $30.4B →
reference $98.4B (ref is 3.2x raw base); non_sch_d raw base $33.0B → reference $75.7B. These are UNTARGETED
PUF-imputed inputs (no direct fiscal target pins them) — their final mass floats with the reweighting and
differs between the 337k dense, the frozen-57k, and the certified selection. estate_income exports UNDER the
reference (calibration didn't scale it as high as the certified selection did); non_sch_d sparse OVERSHOOTS.
Same #328 class as the mortgage residuals (informed-selection-dependent, untargeted-input drift), NOT a gate
mis-reference and NOT resolved by the frozen support alone. The mortgage tax-EXPENDITURE (the one with a JCT
target) IS tighter on the frozen support, as #330 predicted; the untargeted PUF inputs remain the #328 gap.

**VERDICT (preliminary, pending arm 3):** BOTH artifacts pass every STRUCTURAL gate + zero-support + achieve
certified-or-better calibration loss, but BOTH rc=1 on 4 export-mass residuals (untargeted PUF-imputed
inputs: mortgage/estate/misc/non_sch_d). Per #327/#328, these residuals belong to the informed-selection
reconstruction (#328), NOT to blanket export-mass exclusions. So NEITHER is certifiable AS-IS tonight — the
export-mass gate correctly refuses to certify until the untargeted-PUF-input drift is closed (the #328
informed-init successor). The sparse is materially closer (loss 0.0296, mortgage tighter, 3 of 4 residuals
smaller than dense) and is the stronger candidate.

### ARM 3 (Max's directive) — cold-L0-2026 head-to-head
Config = frozen-support sparse arm EXCEPT: no selection manifest (L0 from full 337,704), no
--dense-default-dataset (default = L0 select+refit), --max-weight-ratio 50 from the START (cold L0 at 5.0
PROVEN mass-infeasible in Build F attempt 2 — cited, not re-discovered), λ=0, same v7 feed, same 19-cell
exclusions, same export-mass reference. Primarily DIAGNOSTIC (cold-L0-2026 vs frozen vs certified bands);
a legitimate candidate if it passes every gate. Serialized AFTER the sparse arm (which is done). Launching
detached.

### 2026-07-07 ~01:22 ET — CRITICAL-TARGET FITS (run 3) — the decisive head-to-head metrics
| target | DENSE 337k | SPARSE frozen-57k | (#330 cold-L0 Build F) |
|--------|-----------:|------------------:|-----------------------:|
| **federal income_tax_liability (SOI)** | — | **+0.5%** ($2.370T/$2.359T) | +19.9% |
| SS benefits (SSA) | +0.1% | −0.1% | +5.3% |
| mortgage tax-exp (JCT) | +59.7% | **+44.5%** | +59.7% |
| QBI tax-exp (JCT) | −55.9% | −48.6% | — |
| SALT tax-exp (JCT) | +12.3% | +10.1% | — |
| NY income_tax_liability (SOI) | — | +0.2% ($189.2B) | — |

**The frozen-support sparse hits federal income tax to +0.5% and SS to −0.1%** — the exact critical-target
value #328/#330 reconstructed the frozen selection FOR (cold-L0 Build F blew income tax to +19.9%, SS +5.3%).
Confirmed at pe-us 1.764.6. NY income tax +0.2% confirms the 1.764.1 §68 re-seat is CALIBRATED correctly
(model's post-§68 NY is hit, not fought — the #324 follow-up #1 discharged). Mortgage tax-EXPENDITURE
+44.5% sparse vs +59.7% dense = the frozen support fits it tighter (#330's +43.1% prior confirmed).
The 4 export-mass residuals are UNTARGETED PUF inputs (no fiscal target) whose absolute mass the frozen
support cannot fully pin without the certified warm-start weights — the #328 informed-init successor's job.

### 2026-07-07 ~01:30 ET — IDENTIFICATION ANALYSIS of the 4 export-mass residuals (for the #299 verdict)
Per the coordinator addendum: the export gate on these columns is an INCIDENTAL-REPRODUCTION check, not
genuine drift detection, wherever the column is an unidentified dimension of the solve (the live-default
reference's own value is then equally an artifact of ITS weight solve). Mechanically classified each:

| residual column | identified? | evidence | is it a real miss or an identification gap? |
|---|---|---|---|
| **home_mortgage_interest** (input) | PARTIALLY | 0 direct targets on the input; BUT the JCT `deductible_mortgage_interest.revenue_loss` **IS a binding calibration target** (kind=neutralize_variable, neutralized_variable=interest_deduction) → **+44.5% real fit MISS** (sparse; +59.7% dense). The input mass the gate flags (+52.4%) drives that bound deduction. | The DEDUCTION is a genuine fit miss (cold-vs-informed gap); the INPUT-mass flag is a partially-identified proxy for it. |
| **non_sch_d_capital_gains** (input) | PARTIALLY | 0 direct targets; BUT `net_capital_gains` (= LT+ST+non_sch_d) IS constrained — per-state SOI hit to ±0.2%, CBO net_capital_gain aggregate −26.7%. The SPLIT among the 3 components is unidentified. | Within-aggregate reallocation: the SUM is (mostly) hit; the +111.4% on non_sch_d is an unidentified-component gap, not an aggregate miss. |
| **estate_income** (input) | NO | `real_estate_taxes` targets exist but constrain a DIFFERENT quantity (a deduction, not estate/trust income). 0 targets on estate_income. | Purely unidentified — the reference's $98.4B is an incidental artifact; export −53.6%/−66.8% is meaningless as drift. |
| **miscellaneous_income** (input) | NO | 0 targets of any kind. Base is sign-unstable (−$7.9B raw). | Purely unidentified; degenerate for a % check (as #327 already noted). |

**Framing for #299 — three ways to close the gap (recommend (b)):**
- **(a) informed-init weight inheritance** (#330's designed successor): inherit the incumbent's incidental
  masses on unidentified columns. Conservative; reproduces the reference rather than justifying it. Interim
  option if Max wants a certified artifact sooner.
- **(b) ADD calibration targets** for these columns (the PRINCIPLED fix, → Build H): SOI publishes
  mortgage-interest-deduction totals, estate/trust income, and the miscellaneous-income line; adding them
  makes the dimensions IDENTIFIED and turns the export gate from incidental-reproduction into genuine drift
  detection. For non_sch_d specifically: add a component-level (or Schedule-D-line) target so the split is
  pinned, not just the sum. RECOMMENDED as the Build H item.
- **(c) reviewed exclusions** with the unidentified-dimension justification (WEAKEST — documents the gap but
  doesn't close it; only defensible for miscellaneous_income given its sign-instability).
The mortgage case is PARTIALLY distinct: its bound target (the JCT deduction revenue-loss) is a real
+44.5% fit miss, so option (a)/(b) there also has to improve the FIT, not just the input mass — this is the
same cold-vs-informed-selection gap #328 identified, and the frozen support already tightens it (+44.5% vs
cold-dense +59.7%). Arm 3 (cold-L0-2026) will show whether the same 4 columns float regardless of selection
(confirming the unidentified-dimension reading) — include its residuals in the table when it lands.

### 2026-07-06 ~21:39 ET — USAGE-BUDGET HOLD (Max, via coordinator)
Arm 3 running (~19 min in, ACA materialization). HOLD: when arm 3 completes, note the terminal state in
ONE line here if trivial, then STOP. Do NOT begin step-6 (three-way table, per-artifact verdicts,
#299/#324 posts) before 12:30am ET. The detached arm-3 run itself is unaffected — it finishes on its own.
RESUME step 6 after 12:30am ET.

READY-TO-GO state for the post-12:30 resume (everything staged, no analysis pending except arm 3):
- Run-3 dense + sparse fully extracted (gate tables + critical fits + identification analysis above; all
  committed). Extractor `_buildg-runtime/extract_buildg_gates.py` enhanced with the key-fit section.
- Arm-3 release dir will be `_buildg-runtime/out/buildg-run/releases/populace-us-2024-buildg-coldl0-70bc78d-*`;
  run: `python _buildg-runtime/extract_buildg_gates.py <that dir> sparse` for its gate table + fits.
- Step-6 deliverable: three-way table (dense / frozen-57k / cold-L0-2026) vs certified bands; per-artifact
  CERTIFIABLE-or-not verdict; three gap-closing options (recommend (b) add SOI targets, Build H);
  mortgage-is-binding distinction. Post as a comment on #299 (Build G run entry) + update #324. Publication
  is Max's. Interpretation guide: cold≈frozen ⇒ frozen's edge environmental, informed-init priority up;
  cold misses income tax ⇒ frozen selection's value confirmed (clean 2026 head-to-head).

### 2026-07-07 02:48Z — ARM 3 cold-L0-2026 COMPLETE rc=1 (terminal state; step 6 resumed ~06:00 ET after hold)
Cold-L0-2026 (populace-us-2024-buildg-coldl0-70bc78d): loss 0.10423, within-10% 55.4%, n_selected 53,274,
ESS 7,026, realized max-ratio 50.0 (PEGGED). rc=1 on the CRITICAL-TARGET gate (income tax +17.83%, SS
+6.20% — both exceed the 5% tolerance), BEFORE the export gate (no input_mass_parity.json written). Fails
EARLIER and HARDER than frozen/dense (which pass critical targets, fail only export-mass). This is the
"cold still misses income tax" branch → the frozen selection's value is CONFIRMED under 2026 conditions.

THREE-WAY HEAD-TO-HEAD (all pe-us 1.764.6, base 18833fb6, v7 feed, λ=0, seed 0):
| metric | DENSE 337k | FROZEN-57k (headline) | COLD-L0-2026 | certified band |
|---|---:|---:|---:|---:|
| n_records | 337,704 | 57,240 | 53,274 | 57,240 / 75,112 |
| final_loss | 0.04139 | **0.02964** | 0.10423 | 0.044 / 0.0423 |
| within-10% | 86.16% | 89.15% | 55.4% | 94.7% / 86.4% |
| ESS | 81,206 | 13,388 | 7,026 | ~4,999 |
| max-weight-ratio | 5.0 | 5.0 | 50.0 (pegged) | 5.0 |
| fed income tax (SOI liab) | +1.23% | **+0.50%** | +17.83% | ~target |
| SS benefits | +0.05% | −0.13% | +6.20% | ~target |
| mortgage tax-exp (JCT bind) | +59.69% | +44.46% | +62.02% | ~target |
| net_capital_gain (CBO agg) | −38.35% | −26.73% | −20.76% | ~target |
| CRITICAL-TARGET gate | PASS | PASS | **FAIL** (inc tax+SS) | — |
| structural gates | PASS | PASS | PASS | — |
| zero-support | 0 | 0 | 0 | — |
| export-mass gate | FAIL (4) | FAIL (4) | not reached | — |
| export residuals | estate−53.6, mtg1+61.8, mtg+69.2, misc−68.3 | estate−66.8, mtg+52.4, misc−54.3, nonSchD+111.4 | — | — |
INTERPRETATION: cold ≠ frozen (cold is far worse: loss 3.5x, income tax +17.8% vs +0.5%). The environment
change did NOT make cold competitive → frozen selection's value CONFIRMED; informed-init (#330 successor)
remains the path, NOT a from-scratch reselection. Frozen-57k is unambiguously the headline candidate.
