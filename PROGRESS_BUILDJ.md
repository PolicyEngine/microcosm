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
- **F1 — the base pool does NOT change for assets/SNAP.** #350/#352/#353 (SNAP) and #373 (SCF
  wealth) modified `packages/populace-build/src/populace/build/us_runtime/*` + `source_stages.json`
  + the RELEASE tool `tools/build_us_fiscal_refresh_release.py` — NOT the base builder
  `tools/build_us_puf_support_base.py` (byte-identical to Build F's; `git diff` empty). Assets +
  SNAP are RELEASE-TIME source stages that enrich the frame during the release build. So a base
  rebuild with the same inputs/builder/pe-us is expected to reproduce **18833fb6** (NOT a "new"
  sha). The rebuild is run anyway to (a) satisfy the #368 step and (b) empirically confirm
  determinism; the confirming/again sha is a deliverable. [Base rebuild in flight — Step 1.]
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
- **Step 1 (base rebuild — IN FLIGHT).** Launched `buildj_base.sh` detached (wrapper pid recorded
  in logs/buildj-run/base.wrapperpid). Replicates Build F's exact base command on main's
  (identical) builder: 3-year ASEC pool 2024+2023+2022, 2x PUF clone, seed 0, n-estimators 32,
  aging-facts a5d34d4a, CD assignment (cdx 383a6666, seed 0), block ladder 7ba39b95. Expect sha
  18833fb6 (finding F1). Build F base was 12m14s -> one chunk. [Blocking on base.rc + base.sha.]
