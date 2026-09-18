# Lane report — SPM composition preflight, and the role for a fresh base

Branch `spm-composition-preflight`, cut from `origin/main` at `d1196af10`,
worktree `~/PolicyEngine/_worktrees/microcosm-spm-composition`.
Draft PR: **https://github.com/PolicyEngine/microcosm/pull/948** (never marked
ready, never merged).

> This report is at `experiments/`, not the root `out.md`. `out.md` is a
> tracked file currently holding another lane's committed report (Amendment 19,
> PR #911, last touched 2026-09-12); writing here instead of there avoids
> destroying it.

## 1. What the check does, and where it fires

### The rule it reproduces, read at this head

Installed engine: policyengine-us 2.2.1, spm-calculator 1.0.0,
policyengine-core 3.32.5.

`spm_calculator/policyengine_adapter.py:353-363`

```
spm_measurement_adults = unit.sum((age >= 18) | ((age >= 15) & role))
```

`role` is `is_spm_independent_minor_role`, a Person/ETERNITY bool that carries a
**formula** (`is_household_head | is_household_spouse`, `:342-350`) and is also
the single declared dataset source input
(`policyengine_us.spm.DATASET_SOURCE_INPUTS`). A supplied column therefore wins
and the formula runs only in its absence — pinned upstream by
`policyengine_us/tests/core/test_spm_source_input_contract.py`. With no column,
`is_household_head` and `is_household_spouse` are themselves input-only bools
with no formula of their own, so an absent one contributes `False` and the rule
collapses to `age >= 18`.

`policyengine_amount` (`:300-304`) raises
`SPMInputError("SPM_COMPOSITION_REQUIRED")` on `np.any(adults < 1)` — for the
**whole population**, naming neither the unit nor a remedy.

`spm_calculator/microcosm_adapter.py:219-277` carries the same rule against a
microcosm Frame and names the frame columns the check reads (`PERSON_COLUMNS`).

### `check_spm_composition`

In `us_runtime/release_gate_preflight.py`, shaped like
`check_selection_carryover` and returning a `CheckResult`. It reports:

- the count of SPM units with no classified adult, and the count with no member
  aged 18 or over (the two differ exactly by what the role rescues);
- **the role source** — `source_column`, `household_structure_fallback` (with
  which fallback columns were found), or `unclassified`;
- up to N offending `spm_unit_id`s with each member's age and role, capped at 20
  by default while the failure line still reports the full count;
- missing role values read as False, and ages unreadable as numbers, both
  counted rather than silently absorbed.

`SPM_COMPOSITION_REMEDY` is a module constant, reused verbatim by the release
tool, and it names what is **not** a remedy: re-grouping child-only SPM units
changes the poverty measurement the 104 `in_poverty` rows exist to check, and
inventing an adult is not a fix.

### Where it fires

1. **`run_preflight`** — graded on the *selected* frame when the selection
   mapped, because `attach_l0_refit_entity_weights` calls `base_frame.select`,
   so the export is a strict subset of the pool and a pool unit the selection
   drops never reaches the engine. The pool's own counts ride along in
   `details`. A pool that cannot be classified lands in a SKIPPED branch beside
   the existing ones, so one absent column cannot cost the operator the other
   four checks.
2. **`tools/preflight_us_release_gates.py`** — the CLI renders
   `report.human_table()` generically, so the check needed no rendering work;
   `--max-reported-spm-units` lets an operator widen the named list when
   triaging a fresh lineage.
3. **`tools/build_us_fiscal_refresh_release.py`, hard refusal** on the *export
   frame*, immediately before `_write_reform_validation`. That is the exact
   population: `write_dataset` persists `is_household_head` and
   `is_household_spouse` but **cannot** persist
   `is_spm_independent_minor_role` (the engine adapter marks it formula-owned;
   `PolicyEngineUSEngine.variables()` excludes it — verified: `age` yes,
   `is_household_head` yes, `is_household_spouse` yes, role **no**). Placed
   *after* the batched pre-export raise on purpose: that batching exists so one
   run reports every failing gate at once, and an earlier raise would destroy
   the record it protects.
4. **The same tool, advisory only**, on the base frame before target
   compilation. **I chose not to make this blocking** — since the L0/refit
   export selects a subset, a hard refusal on the pool would reject a run that
   would have succeeded. It buys the pool-level count hours before the solve,
   because a pool that carries the defect needs the role whether or not today's
   selection dodges it. It is wrapped in `except (KeyError, ValueError)` so
   "advisory" is a property of the code, not of a comment.

## 2. The drift guard's result

`packages/microcosm-build/tests/test_us_spm_composition_engine.py`,
`pytestmark = pytest.mark.requires_us`. Every test builds **one** population
spec and derives both the microcosm `Frame` and the policyengine-us situation
from it, so the two can never be handed different populations.

It demonstrates, against the installed engine:

- a minor-only unit → the engine raises `SPM_COMPOSITION_REQUIRED`, and the
  check FAILs naming exactly `spm_unit_id` 2 with member ages `[16.0, 8.0]`;
- the source role on the 15-to-17-year-old → engine accepts, check PASSes;
- `is_household_head` on the minor → both pass via the fallback formula;
- `is_household_spouse` on the minor → both pass;
- a supplied role `False` over a `True` head → **both refuse** (the source
  column wins; this is the inversion that would otherwise pass here and fail in
  the engine);
- and the guard proper: for ages 0/13/14/15/16/17/18/19/64 × role False/True,
  the check's verdict equals the engine's own `spm_measurement_adults` **and**
  equals whether the engine actually raises — comparing against the engine
  rather than against a remembered rule.

`requires_us` is a registered marker (`pyproject.toml`) and the repo-root
`conftest.py` auto-skips it when policyengine-us is absent, so it skips in the
engine-free `fast` lane and runs in the US engine lane.

## 3. The design note's answers

`docs/us-spm-role-for-a-fresh-base.md`.

### The producer is pinned to Build P five ways

| # | Pin | Kind |
|---|---|---|
| 1 | `PARENT_DATASET_SHA256` — parent H5 bytes | evidence about Build P |
| 2 | `PARENT_FILES` — four parent evidence files | evidence |
| 3 | `PARENT_BUILD_ID` — declared identity | evidence |
| 4 | `EXPECTED_COUNTS` — six exact integers | evidence |
| 5 | `REFERENCE_EVIDENCE_SHA256` — the **derived role table's own bytes** | evidence |

Pin 5 is the one that is easy to miss and hardest to generalise: it is not a
parent check. It hashes the CSV the producer just derived and refuses anything
that is not literally Build P's 166,321-row role table, so a *correct*
derivation on any other parent fails there even if pins 1–4 were relaxed.

**Requirements of the method** — already enforced, already parent-agnostic, and
they would survive a generalisation untouched: the parent's required columns and
their integrity; `unmatched_persons == 0`; no source person repeated *within* a
native unit; every native unit drawn from exactly one source unit;
`adult_child_person_count_mismatch_units == 0`; `weights_used` and
`ages_changed` both false; the parent H5 unchanged across the derivation.

**The invariant that is missing today.** Nothing asserts that no unit remains
with zero classified adults *after* the role.
`minor_only_units_resolved = int(units.age_only_adults.eq(0).sum())` counts the
units that **needed** resolving, not the ones that **were** resolved. On Build P
the two coincide, which is why the constant reads as a success number; on a
fresh lineage they need not.

### How the role reaches a person

`SPM_ROLE_RULE = "SPM_HEAD == 1 OR (A_FAMTYP in {1,4} AND A_FAMREL in {1,2})"`
(`spm_role_source.py:24`). It is computed on the **pinned Census ASEC person
CSV** and joined in on `["source_year", "PERIDNUM"]` with
`validate="many_to_one"`. So `SPM_HEAD` never needs to be in the parent; the
parent only needs `source_year` and an exact 22-digit `PERIDNUM`.

Persons with no ASEC origin are **refused, never defaulted**
(`_require(joined._merge.eq("both").all(), "Source join has unmatched parent
persons.")`), which is what `policyengine_us/spm.py` demands when it says source
delivery "does not permit synthesizing a default value when data are absent".

Clones: repetition of a source person **across** native units is allowed
(`many_to_one`); **within** one native unit it is refused; and each native unit
must be one complete Census unit.

## 4. Measured on the phase-2 base

`~/PolicyEngine/_buildq-runtime/out/base-q3/base_populace_us_2024_puf_support.h5`,
read-only. 352,932 households (Path A), 907,382 persons, 367,306 SPM units.
Receipt: `experiments/893-spm-composition-base-q3-receipt.json`; reproduce with
`experiments/spm_composition_base_q3_measurement.py`.

| | |
|---|---|
| SPM units with no member aged 18 or over | **238** |
| …still with no classified adult under this base's own columns | **222** |
| Of the 238, units containing a 15-to-17-year-old | **238 (all)** |
| Members aged 15-17 in those units | 242 |
| Offending unit sizes | 228 singletons, 8 of size 2, 2 of size 4 |
| Their members' channels | 126 `asec`, 126 `puf_tax_detail` |
| `is_household_head` present | yes |
| `is_household_spouse` present | **no** |
| `is_spm_independent_minor_role` present | no |
| `SPM_HEAD` present | **no** |
| `A_FAMTYP` null share | **66.9%** |
| 15-17-year-olds the `A_FAMTYP`/`A_FAMREL` leg alone flags | **10 of 242** |
| Units left unresolved by that leg alone | 230 |
| Persons with a 22-digit `source_person_id` | **907,382 / 907,382** |
| SPM units with no ASEC-origin member | **0** |
| Max repeats of one source person within one SPM unit | 1 |
| `derive_spm_role_source` required columns missing | **none (all 15, incl. `PERIDNUM`)** |
| `source_year` | `{2022, 2023, 2024}` — matches the pins |
| Frame load / check | **8.5 s / 0.05 s** |

Three findings matter most:

1. **A base built from raw sources fails its own release gate today.** 222 is
   not hypothetical.
2. **Every offending unit is resolvable** — all 238 contain a 15-to-17-year-old.
   None is all-under-15, so no unit requires an invented adult.
3. **The base already satisfies every structural precondition of the existing
   derivation**, `PERIDNUM` included. The only thing between it and the role is
   the byte pin.

Two supporting facts:

- The fallback is **head-only** on any microcosm base.
  `with_us_relationship_inputs` emits `is_household_head`, `is_separated` and
  `is_surviving_spouse` and never `is_household_spouse`. At `origin/main`,
  `is_household_spouse` appears in exactly two Python files — the adapter's
  variable table and its test. **It has no producer anywhere in microcosm.**
- The carried raw columns cannot substitute for the rule: `SPM_HEAD` is absent
  and `A_FAMTYP`/`A_FAMREL` are null for 66.9% of persons.

## 5. Recommendation

**Shape (b)** — a source stage beside `relationship_inputs`, reusing
`derive_spm_role_source` unchanged (it already takes `expected_parent_sha256` as
a *parameter*, so the derivation is parent-agnostic today; only the release
packaging is pinned).

Its one blocker is the adapter carve-out, and that carve-out is a **correction**
rather than a relaxation: the `formula_owned` flag records only that the
variable *has* a formula, while `policyengine_us/spm.py` declares
`DATASET_SOURCE_INPUTS = {"is_spm_independent_minor_role"}`, asserts it disjoint
from `REJECTED_DATASET_INPUTS`, and states that "A population producer must
retain its observed boolean instead of treating the fallback formula as
ownership of the input." The rule can read that upstream declaration rather than
hand-maintaining a name. `_GENERATED_VARIABLE_GROUPS` is a generated block, so
the exception belongs beside `_FORMULA_OWNED_COMPAT_COLUMNS`, never as a
hand-edit of the generated table.

**Shape (a) was not implemented, because it does not close** — see §6.

### What it would take to prove on the phase-2 base

Everything structural is proven above. What remains needs the three pinned
Census ASEC person CSVs (gated; not fetched), and is exactly what
`derive_spm_role_source` already measures and fails closed on:

1. `unmatched_persons == 0` across all 907,382 persons;
2. every native SPM unit drawn from one source unit;
3. `adult_child_person_count_mismatch_units == 0`;
4. **the new invariant** — after the role, zero units remain with no classified
   adult, i.e. all 238 *resolved*, not merely counted.
   `check_spm_composition` is the instrument: run it on the enriched frame and
   require PASS.

Steps 1–3 are already-written refusals; step 4 is one assertion. None needs a
decision.

## 6. Questions for Max

**Q. Should the SPM independence role reach a fresh lineage as a build-stage
input leaf, or should the post-hoc enrichment lane be generalised?**

I did not implement shape (a), because it does not close without your call. Not
for the reason the brief anticipated — ASEC-origin coverage is total and no unit
lacks an ASEC-origin member — but for three others:

- pin 5 is an **output** pin, so generalising means dropping an
  independent-oracle guarantee on the derived role table;
- `source_enrichment.py` states directly above `PARENT_FILES` that *"Callers
  cannot grant another schema-5 calibration permission to use the inheritance
  lane by supplying their own pins"* — accepting a declared parent is precisely
  the capability that sentence exists to deny;
- the release type's authority **is** the reviewed parent's byte identity
  ("This release type does not certify a new calibration or upgrade its
  schema"), while a base built from raw sources is calibrated by its own build
  and has no inheritance to claim.

- **Option 1 — (b), my recommendation.** Add the build stage plus the
  `DATASET_SOURCE_INPUTS`-driven adapter carve-out. Build P's lane is untouched
  and stays byte-identical. Cost: one stage shaped like `relationship_inputs`,
  one reviewed adapter exception (and the metadata-index test pin moves with
  it), release input-coverage manifest wiring, and re-pinned spec identities.
  Risk: the release's input surface grows by one person column.
- **Option 2 — (a).** Generalise the enrichment to a declared parent: drop pin
  5, reverse the "callers cannot supply their own pins" intent, and turn
  `EXPECTED_COUNTS` into recorded outputs checked against the §1 invariants plus
  the missing one. Cost: comparable code, plus a governance decision about who
  may declare a parent for the calibration-inheritance lane. Risk: the lane's
  authority is the parent's byte identity, and a fresh base has none to inherit.
- **Option 3 — both, scoped.** (b) for fresh lineages; keep (a)'s lane pinned to
  Build P for descendants of the certified default.

**Nothing in this PR presumes an answer.** The preflight check and the release
refusal are correct under all three options: they name the defect and the
remedy without choosing how the remedy is delivered.

**A second, smaller question.** The pre-calibration point is an *advisory*, not
a refusal, because the L0/refit export selects a subset of the pool, so a
blocking check there could reject a run that would have succeeded. If you would
rather it block — accepting that it can refuse a run whose selection would have
dropped every offender — that is a one-line change from
`_spm_composition_report` to `_assert_spm_composition`.

## 7. Test summaries (verbatim)

Filled in at §8 below.
