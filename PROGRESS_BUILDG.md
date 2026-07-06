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
- [x] 1. Env: `uv sync --all-packages --extra us` (build-g). **RESOLVED pe-us = 1.752.2** (uv.lock honored; sync does NOT bump the pin — that requires `uv lock --upgrade`). Built dedicated venvs `venv-1752` + `venv-1764`, BOTH pinned to **core 3.26.11** (match Build F env) so critical-fit isolates pure pe-us drift. New-pin CEILING under `<2` = **1.765.3** (PyPI latest); issue names 1.764.0 as the window; both reported.
- [x] 2. Fetched + parsed pe-us CHANGELOG 1.752.3→1.765.3 (scratchpad/peus_changelog.md, lines 1-418). Attribution map built (see Log). Confirmed 1.756.9 uprating-placement fix (33 files / 107 nodes ME,NE,ND,NY,OH,RI) + 1.756.8 explicit 2026 values + 1.764.1 NY IT-196 phase-out wiring.
- [~] 3. Task 1 — eCPS parity. KEY MECHANIC: `parity_gate` compares INPUT-COLUMN nonzero shares (`us_nonzero_shares` over base frame) vs frozen JSON reference; base-F frame is FROZEN so the gate result is pin-invariant for stored cols. Pin effect = (a) engine input-var SCOPE shifts, (b) computed-variable drift (state income tax is an OUTPUT). Running BOTH: actual parity gate @ new pin (ratchet) AND estimate-drift table for the 6 named states.
- [ ] 4. Task 2 — take-up contract tests at new pin; per-flag verdict.
- [ ] 5. Task 3 — critical-fit spot checks at 1.752.2 and 1.764.0 on base-F; drift table.
- [ ] 6. Task 4 — new-input inventory (degenerate-gate candidates).
- [ ] 7. Assemble triage report → this file + populace#324 comment. Go/no-go.

## Attribution map (pe-us 1.752.3 → 1.765.3), for tasks 1 & 4
State income tax / parity-relevant:
- **1.756.8** — explicit 2026 values added to NE, ME, RI income tax params; 2024-25 values to NY itemized-deduction phase-out threshold (unfroze via #8905).
- **1.756.9** — moved `uprating:` from sibling-of-`values:` to `metadata:` across **33 param files / 107 nodes**: ME, NE, ND, NY, OH, RI state income tax + RI & NY contrib reforms. Loader had silently ignored them → thresholds/amounts frozen at last explicit value. Adds `test_uprating_placement.py` guard. **← the #324-predicted fix.**
- **1.764.1** — wired NY orphaned itemized-deduction phase-out into IT-196 Line 40 (26 USC 68 overall limitation via NY Tax Law 615) — reduces NY itemized deductions.
- **1.765.2** — CO income tax (199A addback, 2026 fed-deduction-addback exemptions, 2026 CDCC, low-income childcare sunset) [only if pin goes to 1.765.x].
- **1.765.1** — PA retirement-plan distribution exemption + NJ 529 deduction relocation [1.765.x only].
- **1.765.3** — CA AMT itemized double-add fix, AZ property tax credit, LA FITAP / NV TANF rounding [1.765.x only].
Capital-gains / income base (affects income tax + critical-fit):
- **1.756.2** — non-Schedule-D capital gain distributions included in gross income, preferential-rate cap-gains base, and NII.
- **1.756.0** — new inputs `schedule_d_capital_gain_distributions` + `capital_gain_distributions`.
- **1.756.6** — Saver's Credit 2024/25/26 joint AGI thresholds; 2020 UC exclusion AGI fix.
- **1.756.7** — residential clean energy / EE home improvement / clean vehicle credits terminated under OBBBA (Pub. L. 119-21).
OBBBA / federal:
- **1.756.7** clean-energy termination (above); **1.756.1/1.756.4** disabled-adult-dependent CTC/EITC child-count fixes.
SNAP/ABAWD input WAVE (known, task-4 baseline): 1.758.0 (discretionary-exemption PERSON-LEVEL input), 1.760.0 (state-coverage doc), 1.761.0 (former-foster-youth exemption + IHCIA Indian exemption input), 1.762.0 (work-program hours + workfare), 1.763.0 (Alaska borough waivers by county FIPS), 1.764.0 (work-registration exemption hooks: TANF-compliance, drug/alcohol treatment, UC applicants).
NEW inputs BEYOND SNAP wave (task-4 degenerate-gate candidates): `care_expenses` (1.752.3), `capital_gain_distributions` + `schedule_d_capital_gain_distributions` (1.756.0), `employment_income_before_lsr` migration (1.764.3 — 156 fixtures moved off derived `employment_income`; **flag if populace seeds employment_income**), `state_code_str` backfill (1.756.5), `county_fips`→county mapping (1.755.5).

## Log
- **2026-07-06T~1005Z start.** Read #324 + #299 (all 6 comments: Build E terminal result, Build F 5-attempt campaign + attempt-6 pre-flight/close) + #315 (take-up contract) + #316 (parity gate wiring) + #317 (compile survivability). Build F verdict: not certifiable; dense parent one mis-referenced gate from clean (#327), sparse blocked on uncommitted informed-L0 (#328); reusable asset = base `18833fb6`.
- Worktree `build-g` created off origin/main HEAD `e179ba5` (#319 state-legislative validation suite; 3 commits ahead of the ed78097 Build F HEAD). Clean.
- Base-F sha VERIFIED `18833fb6…`. Runtime `_buildg-runtime/` created mirroring `_buildf-runtime` layout.
- pe-us pin `>=1.745.0,<2` (all three `[us]` extras); uv.lock incumbent = 1.752.2; PyPI latest = 1.765.3.
- Parity reference read: 158 vars, incumbent eCPS sha `0a6b961a…` rev `21280dca…`.
- Launched detached `uv sync --all-packages --extra us` (build-g). Writing PROGRESS + committing.
