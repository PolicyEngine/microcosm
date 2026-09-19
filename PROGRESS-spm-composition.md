# SPM composition preflight + role for a fresh base (#893 lane)

Branch `spm-composition-preflight`, cut from `origin/main` at `d1196af10`.

## State

> Historical note (2026-09-18): the state and "Next" sections below describe
> the first session. Part 2's design note shipped in PR #948
> (`docs/us-spm-role-for-a-fresh-base.md`); Max chose its Option 1 (shape b)
> on 2026-09-18. The current state is in "Step 2" at the end of this file.

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

## Next (historical, superseded 2026-09-18)

1. Part 2 design note `docs/us-spm-role-for-a-fresh-base.md`. — shipped in #948.
2. Decide (a) declared-parent generalisation vs (b) source stage; implement (a)
   in this PR if it closes without a decision from Max. — Max chose (b) on
   2026-09-18; implemented on branch `us-spm-role-stage` (see Step 2 below).

# Step 2 — the role as a build-stage input leaf (2026-09-18)

> Historical note (2026-09-19): this section records the earlier session.
> Implementation and actual-wrapper acceptance have since advanced; see the
> dated continuation at the end. Check GitHub for current PR/CI status.

Branch `us-spm-role-stage`, cut from `spm-composition-preflight` at
`0e4b20de7` (PR #948's head, CI green on run 35346454375, mergeable).

## State

Step 1 (make #948 green) was already complete on arrival: the first session's
`0e4b20de7` fixed the seven red jobs' one cause (the gate-failure test's fake
frame lacked a schema; repaired in the fake, not the guard), and the rerun is
green on every job. Step 2 design reading done; proof scripts written; design
note next, then the stage.

## Done

- [x] Verified `gh pr checks 948` all pass on head `0e4b20de7`; mergeable.
- [x] Read at this head: `spm_role_source.py` (byte-identical to main),
      `asec_pool.py`, `relationship_inputs.py`, `education_inputs.py` (sidecar
      pattern), the adapter's two classification paths, the engine's
      `DATASET_SOURCE_INPUTS` declaration, the enrichment lane's pins, the
      coverage manifest generator, the bundle generator's frozen digests, the
      raw ASEC inputs' columns (no vintage carries `SPM_HEAD`; only 2024
      carries `A_FAMTYP`/`A_FAMREL`).
- [x] Located every artifact the proofs need on disk: the three pinned Census
      person CSVs (`~/.cache/microcosm/cps/asec_education/`), the Build P
      parent (HF blob named by its pinned digest), the certified reference
      evidence CSV (digest matches pin 5).
- [x] `experiments/spm_role_stage_proof.py` written (base + buildp receipts).

## Next

1. Run the two proofs (foreground), commit receipts.
2. `docs/us-spm-role-stage.md`.
3. Stage module + manifest + registries + adapter carve-out + coverage manifest
   + export list + base/release tool wiring; regenerate every moved pin through
   its generator; tests; changelog; draft PR.

# Continuation — September 19, 2026

Resumed Claude's `us-spm-role-stage` at `cff157729599c41f7ccfdadf32145d9679ffe796`
and preserved its uncommitted fixture/doc/changelog work. Initial focused
battery: 442 passed.

Independent source review found missing operator-boundary registration and
an impermissible synthesized default for the measured source role. Added
boundary rejection and a regression that failed before the fix. Source inputs
remain exportable but receive no generic default. The simulation projection
refuses an incomplete role, and its audit records 926 inputs / 925 defaults.
Existing role columns are now rederived and compared with the pinned Census
source, rather than trusted because they are nonconstant. Missing provenance
fails the signal gate. Regenerated engine/projection contracts with the
repository generator.

Revised relevant battery: **936 passed**, one country-model divide warning,
167.45 seconds. CI test-group verification passes. Generated bundle/coverage
checks and Fable review are underway at this journal entry.

The new `experiments/spm_role_stage_wrapper_proof.py` invokes the actual stage
and signal gate. Both pinned populations pass: Build P 166,321 persons,
28 unresolved units to zero; base-q3 907,382 persons, 222 to zero. Every role
matches the original derivation; Build P also reproduces the reference CSV
bytes. Every existing column/table/weight/stratum/mass log and every input
file remains unchanged. Receipts record exact source fingerprints and engine
versions. The two original direct-derivation receipts are unchanged.

Remaining: complete generated-artifact checks and review; push a stacked draft
PR; integrate after acceptance. ACS-origin source roles, full-base calibration,
incumbent comparisons and release certification remain open. These proofs do
not establish a new released file.
