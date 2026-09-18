# SPM composition preflight + role for a fresh base (#893 lane)

Branch `spm-composition-preflight`, cut from `origin/main` at `d1196af10`.

## State

Part 1 complete and tested. Part 2 (design note) in progress. PR open as draft.

## Measured on the phase-2 base (read-only, `~/PolicyEngine/_buildq-runtime/out/base-q3/`)

352,932 households / 907,382 persons / 367,306 SPM units.

- 238 SPM units have **no member aged 18 or over**.
- 222 still have **no classified adult** under the engine's actual fallback on
  this base: it carries `is_household_head` but **not** `is_household_spouse`,
  so the fallback is head-only. A release from this base would refuse today.
- Every one of the 238 has a member aged 15-17 (242 of them), so every one is
  resolvable by the role. None is all-under-15.
- `is_spm_independent_minor_role` is absent.
- ASEC origin coverage is **total**: all 907,382 persons carry a 22-digit
  `source_person_id`, including all 474,859 `puf_tax_detail` clones. **Zero** SPM
  units have no ASEC-origin member.
- The raw ASEC columns the base carries cannot supply the rule: `SPM_HEAD` is
  absent, and `A_FAMTYP`/`A_FAMREL` are null for 66.9% of persons. The rule's
  second leg alone resolves only 10 of the 242 15-to-17-year-olds.
- `check_spm_composition` on the real 2.35 GB base: frame load 8.5 s, **check
  0.05 s**, FAIL, 222 units named.

## Verified at this head (not inherited from the brief)

- Installed engine: `policyengine-us 2.2.1`, `spm-calculator 1.0.0`,
  `policyengine-core 3.32.5` (`.venv`, `uv sync --all-packages --locked --extra us`).
- `spm_calculator/policyengine_adapter.py:298-304` — `policyengine_amount`
  reads `spm_measurement_adults` and raises
  `SPMInputError("SPM_COMPOSITION_REQUIRED", ...)` on `np.any(adults < 1)`,
  i.e. for the whole population, not the offending unit.
- `spm_calculator/policyengine_adapter.py:353-363` —
  `spm_measurement_adults = unit.sum((age >= 18) | ((age >= 15) & role))`
  where `role` is `is_spm_independent_minor_role`.
- `spm_calculator/policyengine_adapter.py:342-350` —
  `is_spm_independent_minor_role` is a Person/ETERNITY bool whose **formula**
  is `is_household_head | is_household_spouse`. So a supplied dataset column
  wins; absent one, the formula supplies the head/spouse fallback.
- `tools/build_us_fiscal_refresh_release.py:11731` calls
  `_write_reform_validation` (defined `:7475`) with no handler.

## Done

- [x] Worktree verified at `origin/main` (`d1196af10`), venv synced.
- [x] Engine composition rule read verbatim from installed 1.0.0/2.2.1.
- [x] `release_gate_preflight.py` shape read (`CheckResult`, `PreflightReport`,
      `check_selection_carryover`, `run_preflight` SKIPPED branches).

## Done (Part 1)

- [x] 1.1 `check_spm_composition` + `spm_independence_role` +
      `SPM_COMPOSITION_REMEDY` in `us_runtime/release_gate_preflight.py`.
- [x] 1.2 wired into `run_preflight` (graded on the selected frame, pool
      reported) with a SKIPPED branch, and into the CLI
      (`--max-reported-spm-units`, description, docstring).
- [x] 1.3 `tools/build_us_fiscal_refresh_release.py`: hard refusal on the export
      frame immediately before `_write_reform_validation`; advisory (never a
      raise) on the base frame before target compilation.
- [x] 1.4 `requires_us` drift guard,
      `packages/microcosm-build/tests/test_us_spm_composition_engine.py`
      (23 passed).
- [x] 1.5 13 engine-free tests appended to `test_us_release_gate_preflight.py`
      (55 passed, was 42); `ci_test_groups.py --verify` = ok, new file in
      `us-qs`, not `[defaulted]`; changelog fragment; ruff clean.

## Next

1. Part 2 design note `docs/us-spm-role-for-a-fresh-base.md`.
2. Decide (a) declared-parent generalisation vs (b) source stage; implement (a)
   in this PR if it closes without a decision from Max.
