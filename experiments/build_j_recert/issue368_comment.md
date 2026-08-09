## Build J re-certification run — BLOCKED at the #334 Medicaid take-up gate (fail-closed on a real source hole); asset stage confirmed working en route

Ran the #368 final phase per plan: full gated rebuild on `main` @ `561c198` (#369 coverage gate + reform smoke, #373 SCF asset stage, #384 TANF fix + preflights, #385 release-native loading, SNAP train #350/#352/#353) — base pool rebuild -> rmloss100 selection carry-over -> sparse release with ALL gates on and no bypass flags. **The run stopped exactly where the gate architecture says it must**: the first release-scale execution of #334's Medicaid take-up gate failed closed because Rhode Island has no CMS December-2024 enrollment target — a genuine hole at the CMS source, not an ops failure and not a microcosm defect. STAGING/LOCAL ONLY throughout; nothing touched prod HF or `latest.json`.

Environment: worktree branch `build-j-recert`, policyengine-us **1.764.6** / core **3.26.11** (exact Build I match), seed 0, period 2024, v8 facts `94b7155f`, export-mass reference `c2065b64`, SCF donor `rscfp2022.dta` (zip `3bb4d890` / member `6b8dd2d9`, both matching the #373 pinned digests).

### What completed and passed

| step | result |
|---|---|
| Base pool rebuild (Build F command, main's builder) | **rc=0**, sha `0b50660a21e0…` — base **data identical** to Build F/I's `18833fb6` (summary byte-identical except the self-referential sha+path; `base_household_weight_total` 134,690,323.34024483 exact; builder 0-diff). The sha delta is HDF5 serialization non-determinism, benign. |
| Selection carry-over (#330 identity join, rmloss100 `152baca3`) | **PASS**: n_selected **57,240**, n_unmapped **0**, n_ambiguous **0** — the release selects exactly the Build I support. |
| Registry compile (v8 facts) | **d71c59514e3a / 5,533 specs — identical to Build H/I**, even with #334 in the code (its `medicaid_enrollment` targets were already among the 5,533; the stage is new, the targets are not). Build I comparison stays like-for-like. |
| #373 scf_wealth signal gate | **PASSED** (the run proceeded past its raise-point into Medicaid materialization) — the three SSI countable-resource asset columns (`bank_account_assets`, `stock_assets`, `bond_assets`) materialized **with signal** on the 57k frame. The #368 asset restoration works at release scale. |
| #384 register-consistency + take-up preflights, coverage manifest load | Passed (no raise before the Medicaid stage). |

### The block (verbatim)

```
RuntimeError: Release gates failed: Medicaid take-up failed: states without CMS
enrollment targets: ['44'] — the feed is incomplete; those states would ship
anchored-only enrollment.
```

Run `populace-us-2024-buildj-sparse-rmloss100-01aed9e-20260710T023838Z`, rc=1 at 02:42:52Z, ~4 minutes in, before calibration.

Root cause (full evidence in #386): the CMS PI dataset itself — the pinned ledger artifact — carries **RI 202412 = 0, footnoted "Unable to Provide Data due to System Limitations"** on both the preliminary and final rows. RI did not report December 2024 and CMS never backfilled it (the April 2026 release still carries the 0). The gap is exactly one state-month (RI 202411 = 273,400; 202501 = 279,404). Ledger package and v8 feed are faithful to the source; the registry correctly compiles no spec from a zero count, so 50 of 51 states have targets and RI does not. The gate is unconditionally hard (no bypass flag exists — correct per this issue's own no-bypass doctrine), has no reviewed-exclusion register, and runs before calibration in both arms, so the dense arm would fail identically (not launched; no wasted compute).

This is the #368 gate philosophy vindicated on its first real outing, from an unexpected direction: a merged-but-never-release-tested stage (#334) refused to ship a silently wrong state (RI would have reverted to anchored-only enrollment, invisibly) and instead stopped the release with a precise, diagnosable message.

### Verdict: NOT CERTIFIABLE in this run — blocked, pending a #386 decision (not an artifact-quality failure)

No calibration ran, so there are no Build J loss/within-10%/income-tax/export-mass/SSI-probe numbers to put against the Build I bar (0.030833 / 0.8888 / +0.22% / 0-of-35). The blocker is a one-state data-availability decision that belongs to #334's design owner, deliberately not made here (no synthesized value, no nearest-month splice, no cross-source splice, no invented exclusion). Options are laid out in **#386**: nearest-reported-month anchor, an issue-linked reviewed-exclusion register for unreported states, or a T-MSIS-sourced same-month value.

Resume is mechanical once #386 is decided: all launchers, the carry-over verifier, and the gate extractor are staged and committed on `build-j-recert` (`experiments/build_j_recert/`, runtime mirrors in `_buildj-runtime/`); if the fix changes facts, the relaunch swaps `--ledger-facts` to the re-exported feed and records the new registry version — otherwise it is a stage-side change and the same v8 invariants hold.

Cache-key so far: base `0b50660a` (data == `18833fb6`); pe-us 1.764.6 / core 3.26.11; seed 0; period 2024; feed v8 `94b7155f`; registry `d71c59514e3a` (5,533); selection `152baca3` (57,240; join 0 unmapped / 0 ambiguous); zero-support exclusions `abb106af` (revalidation not reached); export-mass ref `c2065b64`. Working log `PROGRESS_BUILDJ.md` on `build-j-recert`; runtime `~/PolicyEngine/_buildj-runtime/`.
