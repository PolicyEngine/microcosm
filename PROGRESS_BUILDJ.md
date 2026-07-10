# Build J — full re-certification of the whole stack under main's new gates + asset columns

Task: PolicyEngine/populace **#368 (final phase)**. A full gated rebuild proving the whole
stack on `main`: base pool (restored columns) -> dense -> sparse (rmloss100 selection) ->
ALL gates -> verdict on #368. The point vs Build I: the SCF-wealth asset columns (#373) and
Daphne's SNAP-train inputs (#350/#352/#353) now flow, so SSI resource-limit reforms must
score **nonzero**; the #369 coverage/reform-smoke gate, #384 TANF fix + cross-register/take-up
preflights, and #385 release-native loading are all ON.

**STAGING/LOCAL ONLY** — never prod HF / latest.json. Publication + default-flip is Max's call.
Compute discipline: every process < 30 min, chunked, detached (python start_new_session=True +
caffeinate -i + pidfile + rc + append logs), NO killing watchdog (pressure-sample only — six
jetsam kills AND healthy-run watchdog kills taught this). Commit + this log after EVERY step.
Three-strike per chunk. Verbatim numbers only. Never @-mention; --body-file for GitHub bodies.

## Runtime + branch
- Worktree: `/Users/maxghenis/PolicyEngine/_worktrees/populace-build-j-recert` branch
  **build-j-recert** off **origin/main `561c198`** (#385 release-native loading HEAD, which sits
  on #384 #373 #369 + Daphne's SNAP train #350/#352/#353).
- Runtime home (durable, outside repos): `/Users/maxghenis/PolicyEngine/_buildj-runtime/`
  (mirrors `_buildi-runtime` layout: inputs/logs/out/checkpoints/scratch/src).
- venv: worktree `.venv` (uv sync --all-packages --all-extras), policyengine-us **1.764.6** /
  core **3.26.11** (EXACT Build I match); scf_wealth stage imports verified.

## Build I bar (from #359 verdict = the numbers to meet/beat)
final_loss **0.030833** · within-10% **0.8888** · fed income tax (SOI) **+0.22%** ·
export-mass parity **PASS 0/35** · records 57,240 · ESS 12,303 · realized_max_ratio 4.993 ·
zero-support (post-excl) 0. Build I was pinned to Build H lineage and did NOT address #340/#356
(assets absent; SSI reforms scored $0) — that hole is exactly Build J's job.

## Frozen invariants (inherit where identical; the DELTA vs Build I is the source stages + gates)
- base **18833fb6** (expected — see finding F1), pe-us 1.764.6, period 2024, seed 0, lambda=0.
- feed v8 facts **94b7155f** (`consumer_facts_buildh_v8.jsonl`, 131 MB) — REUSED (registry
  identical, finding F2).
- target registry **d71c59514e3a** / 5533 declared -> 5514 sparse after the 19-cell zero-support
  exclusion (finding F2: `fiscal_target_references.json` byte-identical to Build H).
- export-mass reference = live-default 57k **c2065b64** (`populace_us_2024.h5`), +/-50% band, $1B
  floor; reviewed exclusions estate_income + non_sch_d_capital_gains ONLY.
- selection = Build I **rmloss100** manifest (152baca3) carried onto the Build J base (#330
  identity join); 19-cell zero-support exclusions (abb106af) revalidated on Build J.
- SCF donor = `rscfp2022.dta` (Fed econres scfp2022s.zip), zip sha **3bb4d890** / member sha
  **6b8dd2d9** — BOTH match the pinned digests in scf_wealth.py. Pre-fetched to
  `_buildj-runtime/inputs/scf_cache/rscfp2022.dta`; passed via `--scf-summary-extract`.

## Architectural findings (verified on origin/main; correct the task's mental model where needed)
- **F1 — the base pool DATA does NOT change for assets/SNAP; the h5 SHA does (HDF5
  non-determinism).** #350/#352/#353 (SNAP) and #373 (SCF wealth) modified `us_runtime/*` +
  `source_stages.json` + the RELEASE tool — NOT the base builder `build_us_puf_support_base.py`
  (0-diff build-f vs origin/main). Assets + SNAP are RELEASE-TIME source stages. **VERIFIED after
  the rebuild (Step 1):** the Build J base summary is byte-identical to Build F's `18833fb6` base
  summary EXCEPT the self-referential `output_sha256` + `output_h5` path (263/265 json lines
  identical); `base_household_weight_total = 134690323.34024483` matches Build F EXACTLY (a
  QRF/imputation change would perturb it); puf_donor_rows 211677, seed 0, n_estimators 32 all
  match. So the base DATA is unchanged. The base h5 SHA is **`0b50660a…`** (NOT 18833fb6) purely
  because HDF5 serialization is non-reproducible (embedded timestamps / chunk layout), not a data
  change. BENIGN: (a) the checkpoint identity + all arms read the base sha DYNAMICALLY
  (self-consistent on 0b50660a); (b) the Build I comparison stays like-for-like (identical pool
  data); (c) the rmloss100 identity join keys on DATA columns (source_year/household_id/channel/
  clone), not the file hash. The task's "new base sha expected" is literally true (new bytes) but
  the pool is unchanged — corrects my pre-rebuild prediction that the SHA byte-value would reproduce.
- **F2 — the calibration target registry is unchanged.** `fiscal_target_references.json` is
  byte-identical between Build H (b42fbfe) and origin/main (both `010fbfa5…`). So v8 facts
  (94b7155f) remain valid and the registry compiles to the same 5533/5514 surface -> the Build I
  loss/within-10%/income-tax numbers are a like-for-like bar.
- **F3 — the SSI probe reference.** scf_wealth.py's own #368 acceptance probe (seed 0, pe-us
  1.764.6): SCF-only overlay onto the Build H dense frame makes `ssi_countable_resources` nonzero
  for 40.3% of people (dense-native ref 42.5%) and the $10k/$20k reform scores **+$9.7B @ 2026**
  (#374's SCF-only overlay = +$9.66B). This is ABOVE the dense-native +$1.6B/+$16.1B reference
  (restored baseline 4.74M recipients vs 8.05M) for two documented #356 follow-up reasons (SIPP
  blend absent — the SCF draw omits the low liquid-asset bottom mass; and the overlay is on an
  already-calibrated frame). The **re-solved rebuild should land LOWER than +$9.66B** (task
  expectation); the SIPP blend is a KNOWN open refinement (#356), to be stated, not fixed here.

## TIMELINE
### 2026-07-09/10
- **Step 0 (setup + recon — DONE).** Read #368, the #359 Build I verdict (bar above),
  PROGRESS_BUILDH/I.md, and the codex architecture review §3 (release-tool failure modes:
  non-transactional local construction; frame + per-reform vectors are the only restartable
  products with atomic writes; source stages rerun until the frame checkpoint hits; no optimizer
  checkpoint; the warm-start-into-L0 config trap; memory — the 88 GB risk is the full-pool solve,
  sparse peak RSS ~3.6 GB). Created worktree `build-j-recert` @ 561c198 + runtime `_buildj-runtime`.
  Verified inputs: census_cps 2024/2023/2022 + puf_2024 present; base-build aux inputs match Build
  F shas (aging-v5 a5d34d4a, cdx 383a6666, bladder 7ba39b95); v8 facts 94b7155f present. Pre-fetched
  + sha-verified the SCF 2022 summary extract (zip 3bb4d890, member 6b8dd2d9 — both == pinned).
  Established findings F1/F2/F3 above. Wrote the detacher (`detach.py`, start_new_session=True +
  caffeinate) and the base launcher (`buildj_base.sh`, integrity preflight + pressure sampler, no
  killing watchdog).
- **Step 1 (base rebuild — DONE, rc=0).** `buildj_base.sh` completed 02:06:03Z (~19.6 min wall;
  Build F was 12m14s — the extra is CPU contention with concurrent recon, healthy throughout: 99%
  CPU, RSS peaked ~53 GB, free% ~86, no jetsam). Base at
  `_buildj-runtime/out/base-j/base_populace_us_2024_puf_support.h5` (1,879,908,154 bytes). **Base
  sha `0b50660a21e0e138cbdcf941303240435b8cf92f4ca3824b2c062ae1085e9c51`** — differs from Build F's
  18833fb6, but the base DATA is IDENTICAL (finding F1: summary byte-identical except the self-sha +
  path; base_household_weight_total 134690323.34024483 == Build F exactly; builder 0-diff build-f
  vs main). The SHA delta is HDF5 serialization non-determinism, not a data change. Proceeding on
  0b50660a (read dynamically by every arm).
- **Step 2 (selection carry-over — running).** `verify_selection_carryover_buildj.py`: #330 identity
  join of the Build I rmloss100 manifest onto base-j in frozen_support mode (raises on any
  unmapped/ambiguous = the >0-miss STOP). Expect n_selected 57,240, n_unmapped 0, n_ambiguous 0
  (the join keys on data-identity columns, unaffected by the h5-byte delta).

## Gate map + verdict-data locations (verified on origin/main; for the resumed run + verdict)
Runs = worktree `.venv/bin/python tools/build_us_fiscal_refresh_release.py`; the sparse (deployable)
release dir holds every certification artifact. NO bypass flags are passed (gates ALL ON).
- **calibration_diagnostics.json** — final_loss, within-10%, ESS, realized_max_weight_ratio, key
  fits (fed income tax SOI, SS, net cap gain, mortgage tax-exp). Bar = Build I 0.030833 / 0.8888 /
  +0.22%.
- **input_mass_parity.json** — export-mass gate (35 cols, ±50%, $1B floor; reviewed exclusions
  estate_income + non_sch_d_capital_gains ONLY). Bar = Build I PASS 0/35.
- **reform_coverage_smoke.json** — the #369 SSI probe `ssi_asset_limit_10k_20k` (binding on
  bank_account_assets/stock_assets/bond_assets; individual->$10k couple->$20k; direction
  reform_minus_baseline; **PERIOD = 2024**; $1B floor). Read `details.results.ssi_asset_limit_10k_20k
  .effect` — VERBATIM = the SSI probe magnitude for the verdict. Gate hard-fails if <$1B (no bypass).
  Note the release scores at **2024** (the scf_wealth doc's +$9.7B / #374's +$9.66B SCF-only overlay
  were quoted at 2026 — state the period alongside the number). A relaxation raises SSI cost -> effect
  positive. Expected: nonzero (>=$1B), LOWER than the +$9.66B overlay (re-solve vs overlay), still
  above the dense-native +$1.6B (the SIPP-blend gap, #356 open — state, don't fix).
- **Structural + coverage + degenerate + take-up + register gates** — the release raises/writes a
  verdict per gate; `release_gates.passed` is the aggregate. Watch surfaces:
  - **#369 input-coverage manifest** (`release_input_coverage_manifest.json`: 65 required + 93
    reviewed-exclusions incl. the 3 asset leaves now REQUIRED). Hard, no `--allow-input-coverage-gaps`.
  - **#369 reform-smoke** (above).
  - **degenerate (#286)** — TANF exclusion REMOVED by #384 -> `takes_up_tanf_if_eligible` must now
    carry signal AND be unexcused (Build I's rmloss100 made it non-degenerate). A constant-at-default
    unexcused column fails; a stale exclusion fails.
  - **#384 register-consistency (#377)** — cross-checks signal-side vs excused-side registers; any
    column both required-to-signal AND excused (degenerate/documented-absent) fails. Cheap preflight.
  - **#384 cross-register + take-up preflights**; **eCPS parity (#316)**; **export-mass**; **base
    population / immigration (#266) / validation-input-coverage (#278/9)** structural gates.
  A gate failure is a FINDING: diagnose precisely (three-strike per chunk), do NOT bypass.
- **release_manifest.json** + exported **populace_us_2024.h5** — written last; `--skip-reform-validation`
  avoids the Build I MD-UNKNOWN tail bug (policyengine-us#8975 / populace#367) without touching the
  certified dataset (gates + export-mass + H5 all complete before the optional reform-validation tail).
