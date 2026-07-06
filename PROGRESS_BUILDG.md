# Build G — pin-bump validation pass (pe-us 1.752.2 → 1.764.x) — PROGRESS

Task: PolicyEngine/populace **#324** (Build G opener). Quantify pe-us model drift across the
Build F release window (1.752.2 → 1.764.0, 28 releases) BEFORE Build G commits to a pin.
**Validation-only pass. No build runs, no register commits, no pushes to main.**

Worktree: `/Users/maxghenis/PolicyEngine/_worktrees/populace-build-g` (branch `build-g` off `origin/main`).
Runtime home (OUTSIDE repo): `/Users/maxghenis/PolicyEngine/_buildg-runtime/` (inputs/ logs/ checkpoints/ venvs/ out/ parked/).
All long compute detached (nohup + pidfile + log). PROGRESS committed every step.

## Fixed inputs (verified, do NOT rebuild)
- **Base F artifact** (REUSE): `~/PolicyEngine/_worktrees/populace-build-f/out/base-f-20260705/base_populace_us_2024_puf_support.h5`
  - sha256 `18833fb68e60ee74461608d81a5c5ab7d52435e17026d9e3b062d9de18d6871f` — **VERIFIED** matches required `18833fb6…`.
  - 3-year ASEC pool (2024+2023+2022 equal thirds) → 2× PUF clone = 168,852 hh (person 432,523; tax_unit 231,007). base weight total 134.69M (pre-334.2M release rescale).
  - All four #278 leaves present with mass.
- **Incumbent pin (Build F / uv.lock)** = policyengine-us **1.752.2**, core 3.26.11.
- **New pin**: `>=1.745.0,<2` resolves upward. PyPI latest at run time = **1.765.3**. Issue names the window ceiling as **1.764.0**.
  - DECISION: record the natural resolution (build-g worktree sync), AND build a dedicated **1.764.0** venv so the critical-fit drift is measured over exactly the issue's named window (1.752.2 → 1.764.0). If natural resolution is 1.765.x, report both and note 1.765 delta separately.

## The four validation tasks (#324)
1. **eCPS parity gate (ratchet mode)** at the new pin vs pinned incumbent reference
   (`packages/populace-build/src/populace/build/us/ecps_parity_reference.json`, 158 vars, pinned to incumbent
   eCPS sha `0a6b961a…`). Expected legit deltas: state income tax for ME/NE/ND/NY/OH/RI from 1.756.9
   uprating-placement fix (33 frozen param files now uprate) + 1.756.8 explicit 2026 values.
   Deliverable: per-variable delta table (variable, old est, new est, delta, attributable-release+changelog cite,
   PROPOSED register disposition). **Proposals only — lead gates them, no register commits.**
2. **Take-up contract assertion** (#315 engine-asserted contract, `us/take_up_contract.json` +
   `us_runtime/take_up_contract.py`): run take-up contract tests at new pin; confirm no model-side seeding
   appeared for data-seeded flags and defaults didn't shift semantics. Pass/fail per flag (13 flags, all
   data_seeded at 1.752.2).
3. **Critical-fit spot checks**: compute federal income tax liability (amount/returns), taxable SS
   (amount/returns), EITC amount, ACTC claims on the base at BOTH pins (two small venvs). Pure model drift
   BEFORE calibration.
4. **New-input inventory**: scan 1.753–1.764 changelogs for anything touching populace's input surface
   beyond the known SNAP/ABAWD wave. New person-level inputs consuming engine defaults = future
   degenerate-gate candidates.

## Deliverable
Triage report → committed to this file + posted as ONE comment on populace#324: drift tables, per-delta
attributions, register-entry proposals, take-up verdict, new-input inventory, go/no-go for pinning 1.764.x.

## Steps
- [x] 0. Read #324, #299 (Build E/F campaign comments), #315/#316/#317. Set up worktree, verify base sha, mirror runtime layout.
- [~] 1. Env: `uv sync --all-packages --extra us` in build-g worktree (detached). Record resolved pe-us. Build dedicated 1.752.2 + 1.764.0 venvs for the two-pin critical-fit.
- [ ] 2. Fetch + parse pe-us CHANGELOG 1.753–1.764: build the release→change attribution map (state-tax uprating fix, SNAP/ABAWD wave, new inputs).
- [ ] 3. Task 1 — eCPS parity: compute base nonzero shares at new pin, diff vs pinned reference (ratchet mode). Per-variable delta table + attributions + PROPOSED dispositions.
- [ ] 4. Task 2 — take-up contract tests at new pin; per-flag verdict.
- [ ] 5. Task 3 — critical-fit spot checks at 1.752.2 and 1.764.0 on base-F; drift table.
- [ ] 6. Task 4 — new-input inventory (degenerate-gate candidates).
- [ ] 7. Assemble triage report → this file + populace#324 comment. Go/no-go.

## Log
- **2026-07-06T~1005Z start.** Read #324 + #299 (all 6 comments: Build E terminal result, Build F 5-attempt campaign + attempt-6 pre-flight/close) + #315 (take-up contract) + #316 (parity gate wiring) + #317 (compile survivability). Build F verdict: not certifiable; dense parent one mis-referenced gate from clean (#327), sparse blocked on uncommitted informed-L0 (#328); reusable asset = base `18833fb6`.
- Worktree `build-g` created off origin/main HEAD `e179ba5` (#319 state-legislative validation suite; 3 commits ahead of the ed78097 Build F HEAD). Clean.
- Base-F sha VERIFIED `18833fb6…`. Runtime `_buildg-runtime/` created mirroring `_buildf-runtime` layout.
- pe-us pin `>=1.745.0,<2` (all three `[us]` extras); uv.lock incumbent = 1.752.2; PyPI latest = 1.765.3.
- Parity reference read: 158 vars, incumbent eCPS sha `0a6b961a…` rev `21280dca…`.
- Launched detached `uv sync --all-packages --extra us` (build-g). Writing PROGRESS + committing.
