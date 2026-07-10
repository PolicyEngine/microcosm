## The failure (Build J, first release-scale run of #334)

The Build J re-certification sparse release (#368, run `populace-us-2024-buildj-sparse-rmloss100-01aed9e-20260710T023838Z`, commit 01aed9e on origin/main 561c198) fails closed at the #334 Medicaid take-up gate:

```
RuntimeError: Release gates failed: Medicaid take-up failed: states without CMS
enrollment targets: ['44'] — the feed is incomplete; those states would ship
anchored-only enrollment.
```

This is the gate working exactly as #334 designed it ("fails on … a state missing its target row") — its 23 tests covered the missing-target mode on fixtures, but Build J is the first full-release run against the real v8 feed, and the real feed has exactly one hole: Rhode Island.

## Root cause: RI's December 2024 CMS snapshot is unreported at source

- **CMS PI dataset** (`pi-dataset-april-2026-release.csv`, the pinned ledger artifact): both RI 202412 rows — preliminary (P/N) and final (U/Y) — carry `Total Medicaid Enrollment = 0` with footnote **"Unable to Provide Data due to System Limitations"**. RI did not report December 2024. Every other 202412 state row has a real count.
- **The gap is exactly one state-month.** RI's own series is otherwise continuous: 202411 = 273,400 and 202501 = 279,404 (final U/Y rows in the same pinned CSV). CMS never backfilled it — the April 2026 release, 16 months later, still carries the footnoted 0.
- **Ledger package is faithful, not defective**: `cms-medicaid-chip-monthly-enrollment-december-2024` (vintage `april_2026_release`, extracted 2026-05-11) selects RI's 202412 U/Y row per spec; the extraction correctly mirrors the source's 0.
- **v8 feed is faithful**: `consumer_facts_buildh_v8.jsonl` (94b7155f) carries all five RI cms_medicaid facts at month 2024-12 with value 0 (e.g. `cms_medicaid.month2024_12.state_enrollment.ri.total_medicaid_enrollment`).
- **Registry compile drops the zero**: `compile_us_fiscal_target_registry` on v8 yields registry `d71c59514e3a` / 5,533 specs (identical to Build H/I — the medicaid_enrollment targets were already among them) with **50 state-level `medicaid_enrollment` specs; FIPS 44 absent** (a 0-valued count compiles to no spec). `_medicaid_source_target_table` therefore has no RI row, and `us_medicaid_take_up_gate` fails closed.

So: the stage has no RI target because no true RI December-2024 count exists anywhere in the pinned source. Re-exporting the feed cannot fix this; the hole is at CMS.

## Why this blocks Build J (and any future certification on main)

The gate is unconditionally hard — no `--allow-*` bypass exists for it (correct per #368's no-bypass doctrine), the stage has no reviewed-exclusion mechanism, and it runs before calibration in both dense and sparse arms. Every release build on current main fails at this gate until RI has either a target or a documented exception.

Notably, the run got past the #373 scf_wealth signal gate before failing here — the SSI asset columns materialized with signal on the 57k frame. The #368 asset restoration itself is on track; this is a separate, newly exposed data hole.

## Decision needed (data-design call, deliberately not made in Build J)

Build J stopped rather than synthesize a value. Options, for whoever owns #334's design:

1. **Nearest-reported-month anchor for unreported state-months**: target RI at 202411 (273,400) or 202501 (279,404), tagged in provenance as a substituted month. Smallest change; slightly breaks the uniform point-in-time month (#332) for one state.
2. **Reviewed-exclusion register for unreported states**: let RI ship anchored-only with an issue-linked exclusion, mirroring the #286 cannot-rot semantics. Preserves month purity; RI's enrollment then rests on the CPS anchor floor alone (undercount known).
3. **Alternative CMS source for the same month**: the T-MSIS-based monthly enrollment snapshot reports RI December 2024 independently of the PI dataset. Cross-source consistency with the other 50 states needs review before splicing.
4. Any of the above requires a feed re-export only if the chosen fix changes the facts (option 1 or 3); option 2 is stage-side only.

## Evidence trail

- Failed run log: `_buildj-runtime/logs/buildj-run/release_sparse.log` (rc=1, 02:42:52Z)
- CSV rows: `ledger` repo `db/data/cms_medicaid/chip_monthly_enrollment_dataset/pi-dataset-april-2026-release.csv`, RI 202412 P/N + U/Y rows
- Package: `ledger` repo `packages/cms_medicaid/chip_monthly_enrollment_december_2024/source_package.yaml` (RI selected-row spec at lines 175-178)
- Registry verification: 50/51 state specs, RI absent, version d71c59514e3a (Build J worktree, pe-us 1.764.6, seed 0)

Refs: #334 (the stage + gate), #368 (Build J re-certification, blocked on this), #332 (point-in-time month doctrine), #321 (the CHIP analogue of a concept-level data gap).
