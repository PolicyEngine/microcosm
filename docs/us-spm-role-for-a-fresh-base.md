# The SPM independence role for a base that is not Build P

Every claim here was read at `spm-composition-preflight`'s head, which is
`origin/main` `d1196af10` plus this branch. The installed engine is
policyengine-us 2.2.1 / spm-calculator 1.0.0 / policyengine-core 3.32.5.

## The problem in one paragraph

`spm-calculator` 1.0.0 refuses the **whole population's** SPM measurement if
*one* SPM unit has no classified adult
(`spm_calculator/policyengine_adapter.py` `policyengine_amount`,
`if np.any(adults < 1): raise SPMInputError("SPM_COMPOSITION_REQUIRED", ...)`).
A unit is classified by
`spm_measurement_adults = unit.sum((age >= 18) | ((age >= 15) & role))`, where
`role` is `is_spm_independent_minor_role`. The certified default passes only
because of a post-hoc enrichment that adds that column to **Build P's exact
bytes**. A base built from raw sources today carries no such column, so it
cannot pass its own release gate. Measured on the phase-2 base: **222 SPM units
have no classified adult**, and a release from it would refuse.

## 1. What ties the producer to Build P

Five independent pins. None of them is an override-able flag; there is no escape
hatch anywhere in the producer or the validator.

| # | Pin | Declared at | Enforced at |
|---|---|---|---|
| 1 | `PARENT_DATASET_SHA256` — the parent H5's bytes | `source_enrichment.py:56` | `build_us_spm_role_enrichment.py:117-118` (`"Only the exact reviewed BuildP parent can be enriched"`), then re-asserted as `expected_parent_sha256` into `derive_spm_role_source` (`:128`) and `append_native_spm_role` (`:158`), and again by the validator |
| 2 | `PARENT_FILES` — four parent evidence files' SHA-256 | `source_enrichment.py:61` | `build_us_spm_role_enrichment.py:119-122` and, after copying, in the validator |
| 3 | `PARENT_BUILD_ID` — the parent's declared release id | `source_enrichment.py:55` | echoed into the report's `parent` block and the build manifest's inherited-calibration block |
| 4 | `EXPECTED_COUNTS` — six integers | `source_enrichment.py:111` | `_check_source_provenance`, strict equality with `type(...) is not int` |
| 5 | `REFERENCE_EVIDENCE_SHA256` — the **derived role table's own bytes** | `build_us_spm_role_enrichment.py:37` (as `SOURCE_EVIDENCE_SHA256`, `source_enrichment.py:31`) | `build_us_spm_role_enrichment.py:139-141`: `"Reconstructed source evidence differs from reviewed BuildP roles"` |

Pin 5 is the one that is easy to miss and the hardest to generalise: it is not a
parent check at all. It hashes the CSV the producer just derived and refuses
anything that is not literally Build P's 166,321-row role table. A *correct*
derivation on any other parent fails there even if pins 1–4 were relaxed.

`EXPECTED_COUNTS` verbatim:

```python
EXPECTED_COUNTS = {
    "persons_joined": 166321,
    "native_spm_units": 59900,
    "total_source_people": 432523,
    "total_source_units": 176039,
    "minor_only_units_resolved": 28,
    "classification_changed_units_vs_age_only": 132,
}
```

### Evidence about Build P, versus requirements of the method

**Requirements of the method** — these must hold for *any* parent, and the code
already fails closed on each. They are not Build-P-specific and would survive a
generalisation untouched:

- The parent carries the required identity/age/count columns and none of them is
  missing (`spm_role_source.py`, the `required` tuple and `_require_columns`).
- `person_id` and `spm_unit_id` unique; SPM membership exactly covers the SPM
  table.
- The parent's income years, the pinned Census sources and the supplied CSV paths
  are the same set.
- **`unmatched_persons == 0`** — every parent person joins a pinned ASEC row.
- Every raw field the parent carries agrees with the pinned Census source
  (`_REQUIRED_RAW_CHECKS`, and `_OPTIONAL_RAW_CHECKS` where present).
- No source person repeats **within** one native SPM unit.
- Every native SPM unit is drawn from exactly one source unit.
- **`adult_child_person_count_mismatch_units == 0`** — the derived adult/child
  counts reconcile against the source's own `SPM_NUMADULTS` / `SPM_NUMKIDS`.
- `weights_used is False`, `ages_changed is False`.
- The parent H5 is unchanged between the start and end of the derivation.

**Evidence about Build P** — true measurements of one population, which a
generalisation would have to turn into *recorded outputs*:

- Pins 1, 2, 3 and 5, and all six `EXPECTED_COUNTS` values.
- Two of the six (`persons_joined`, `native_spm_units`) are additionally
  recomputed from the candidate's own evidence CSV and H5 by
  `_check_person_evidence` and cross-checked against the provenance. That
  *self-consistency* arm is a requirement and survives; only the Build-P values
  are evidence.

**A requirement that is missing today.** Nothing asserts that *no unit remains
with zero classified adults after the role is applied*.
`minor_only_units_resolved` is `int(units.age_only_adults.eq(0).sum())` — the
number of units that had no member aged 18 or over, i.e. how many *needed*
resolving, not how many *were* resolved. On Build P the two coincide, which is
why the constant reads as a success number. On a fresh lineage they need not,
and this is exactly the invariant a declared-parent generalisation would have to
add.

## 2. How the role reaches a person

```
SPM_ROLE_RULE = "SPM_HEAD == 1 OR (A_FAMTYP in {1,4} AND A_FAMREL in {1,2})"
```
(`spm_role_source.py:24`.)

The role is **not** computed from columns the parent carries. It is computed on
the pinned Census ASEC person CSV and joined in:

```python
joined = parent.merge(
    source, on=["source_year", "PERIDNUM"], how="left",
    sort=False, validate="many_to_one", suffixes=("_parent", ""), indicator=True,
)
```

So the parent's only obligation is to carry `source_year` and an exact
22-digit `PERIDNUM`. `SPM_HEAD` itself never has to be in the parent — and in
the phase-2 base it is not.

**Persons with no ASEC origin are refused, never defaulted.**
`_require(joined._merge.eq("both").all(), "Source join has unmatched parent
persons.")`. This is what `policyengine_us/spm.py` requires when it says the
source declaration "permits source delivery; it does not permit synthesizing a
default value when data are absent" — there is no branch that invents a role.

**Clones.** The module's own docstring says support clones may repeat a source
person across native units. `validate="many_to_one"` allows exactly that:
repetition **across** native units is fine, repetition **within** one native unit
is refused, and each native unit must be one complete Census unit (checked by the
single-`SPM_ID` test and by `_reconcile_units`' count reconciliation).

## 3. The phase-2 Path A base, measured

`~/PolicyEngine/_buildq-runtime/out/base-q3/base_populace_us_2024_puf_support.h5`,
read-only — 352,932 households, 907,382 persons, 367,306 SPM units. Reproduce
with `experiments/spm_composition_base_q3_measurement.py`.

### The defect

| | |
|---|---|
| SPM units with no member aged 18 or over | **238** |
| …still with no classified adult under this base's own columns | **222** |
| Of the 238, units containing a 15-to-17-year-old | **238 (all)** |
| Members aged 15-17 in those units | 242 |
| Unit sizes | 228 singletons, 8 of size 2, 2 of size 4 |
| Support channels of their members | 126 `asec`, 126 `puf_tax_detail` |

Every offending unit contains a 15-to-17-year-old, so **every one is in principle
resolvable by the role**. None is all-under-15, and none needs an invented adult.

### Why the fallback does not save it

The engine's fallback formula is `is_household_head | is_household_spouse`. This
base carries `is_household_head` but **not** `is_household_spouse`, and neither
does any other microcosm base: `with_us_relationship_inputs` emits
`is_household_head`, `is_separated` and `is_surviving_spouse` and never
`is_household_spouse` (`US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS`). At
`origin/main`, `is_household_spouse` appears in exactly two Python files — the
engine adapter's variable table and its test. **It has no producer anywhere in
microcosm.** So the fallback is head-only, and it leaves 222 of the 238.

### Why the carried raw columns cannot substitute

`SPM_HEAD` is absent from this base, and `A_FAMTYP` / `A_FAMREL` are null for
**66.9%** of persons (equally in both channels). The rule's second leg alone
flags only **10 of the 242** 15-to-17-year-olds and leaves 230 units unresolved.
The `SPM_HEAD` leg is not optional in practice.

### The base already meets every structural precondition

All fifteen columns `derive_spm_role_source` requires are present, including
**`PERIDNUM`**; `source_year` is exactly `{2022, 2023, 2024}`, matching the
pinned `ASEC_SPM_ROLE_SOURCES`; `person_id` and `spm_unit_id` are unique; SPM
membership exactly covers the SPM table; the required raw fields have no missing
values; and no source person repeats within a single SPM unit.

### The brief's open question, answered

> whether a fresh Path A base has any SPM unit whose members all lack an ASEC
> origin

**No.** All 907,382 persons carry a 22-digit `source_person_id` — including all
474,859 `puf_tax_detail` clones — and **zero** SPM units lack an ASEC-origin
member. The "no synthesizing defaults" constraint is satisfiable on this base
without inventing anything.

## 4. The two candidate shapes

### (a) Generalise the enrichment to accept a declared parent

Mechanically this is close: `derive_spm_role_source` already takes
`expected_parent_sha256` as a *parameter*, so the derivation is parent-agnostic
today. Only the release packaging is pinned.

It does not close, for three reasons, in increasing order of weight:

1. **Pin 5 pins the output, not the parent.** Generalising requires dropping a
   byte-equality check on the derived role table, which is an independent-oracle
   guarantee, not a parent identity.
2. **The module says callers must not be able to do this.** `source_enrichment.py`,
   immediately above `PARENT_FILES`: *"Reviewed immutable BuildP evidence. Callers
   cannot grant another schema-5 calibration permission to use the inheritance
   lane by supplying their own pins."* Accepting a declared parent is precisely
   the capability that comment exists to deny.
3. **A fresh lineage has nothing to inherit.** The release type's own docstring:
   *"This release type does not certify a new calibration or upgrade its schema.
   Its authority is the reviewed parent byte identity, an exhaustive H5
   comparison, Census source reconciliation…"* Its authority **is** the parent's
   byte identity. A base built from raw sources is calibrated by its own build;
   pointing the calibration-inheritance lane at it would claim an inheritance
   that does not exist.

Shape (a) therefore turns a narrow, reviewed, byte-pinned lane into a general
one. That is a governance change, not an engineering one — see §6.

### (b) A source stage beside `relationship_inputs`

Derive the role during the build and emit it as an input leaf, exactly as
`relationship_inputs` emits `is_household_head` from `P_SEQ == 1`.

The derivation does not need to be rewritten: `derive_spm_role_source` already
does it, already fails closed on every method requirement, and already takes the
parent digest as a parameter. The stage would wrap it against the base frame
rather than against a published parent.

One real blocker, and it is the one the brief anticipated. The engine adapter's
generated-variable audit marks the role **formula-owned**:

```python
(
    ("is_spm_independent_minor_role",),
    "person", "bool", "point",
    True,          # <- formula_owned
),
```

so `PolicyEngineUSEngine.variables()` excludes it and `write_dataset` refuses it
through `_engine_computed_columns`. Verified at this head:

| variable | exportable |
|---|---|
| `age` | yes |
| `is_household_head` | yes |
| `is_household_spouse` | yes |
| `is_spm_independent_minor_role` | **no** |

**The carve-out is principled, not an exception.** The `formula_owned` flag
records only that the variable *has* a formula. The engine's own contract says
the opposite about ownership: `policyengine_us/spm.py` declares
`DATASET_SOURCE_INPUTS = frozenset({"is_spm_independent_minor_role"})`, asserts
it disjoint from `REJECTED_DATASET_INPUTS`, and comments that *"A population
producer must retain its observed boolean instead of treating the fallback
formula as ownership of the input."* `is_spm_independent_minor_role` is also
**not** in `FORMULA_OWNED_INPUTS`. So microcosm's classification contradicts
upstream's explicit declaration, and the carve-out can read that declaration
rather than hand-maintaining a name: *a variable upstream lists in
`DATASET_SOURCE_INPUTS` is source-deliverable.*

Note that `_GENERATED_VARIABLE_GROUPS` is a generated block
(`tools/refresh_us_generated_variable_audit.py`), so the carve-out belongs beside
`_FORMULA_OWNED_COMPAT_COLUMNS` as a declared, reviewed exception — never as a
hand-edit of the generated table.

### The shape that is *not* a remedy

Adding an `is_household_spouse` producer would work through the existing fallback
with no adapter change at all. It should not be done as a substitute for the
role. The fallback measures *household headship*; the role measures *SPM-unit
independence*, and the two differ on 132 units on Build P
(`classification_changed_units_vs_age_only`). Delivering headship under the
role's name is exactly what `policyengine_us/spm.py` warns against.

## 5. Recommendation

**(b), reusing `derive_spm_role_source` unchanged as the derivation.**

It is the only shape that gives a fresh lineage the *observed* role, it needs no
claim of inherited calibration, and its one code change corrects a
classification that already disagrees with the engine's own contract.

Cost, as read:

- A `with_us_spm_independence_role` stage in `us_runtime`, shaped like
  `relationship_inputs`: a manifest stage spec, a signal gate, a summary, and a
  `_relationship_surface_carries_signal`-style idempotence guard. The derivation
  body is a call into `derive_spm_role_source`.
- The adapter carve-out plus its metadata-index test
  (`test_policyengine_us_metadata_index.py` currently pins `formula_owned=True`
  for this name, so that pin moves with it).
- Wiring in `tools/build_us_fiscal_refresh_release.py` beside the other
  `with_us_*` stages, and the release input-coverage manifest entry.
- Spec-identity consequences: adding an attested module moves the identity
  artifacts; re-pin rather than hunt for an environment leak.
- The existing Build P enrichment lane is untouched and stays byte-identical.

### What it would take to prove on the phase-2 base

Everything **structural** is already proven above: columns, years, uniqueness,
membership coverage, no within-unit source repeats, and total ASEC-origin
coverage.

What remains needs the three pinned Census ASEC person CSVs (gated; not fetched
here), and is exactly what `derive_spm_role_source` already measures and fails
closed on:

1. `unmatched_persons == 0` for all 907,382 persons.
2. Every native SPM unit drawn from one source unit.
3. `adult_child_person_count_mismatch_units == 0`.
4. **The new invariant**: after the role, zero units remain with no classified
   adult — i.e. all 238 are resolved, not merely counted. The check this branch
   adds (`check_spm_composition`) is the instrument: run it on the enriched
   frame and require PASS.

Steps 1–3 are already-written refusals; step 4 is one assertion. None of them
needs a decision.

## 6. The question for root

Shape (b) is recommended and does not require relaxing anything. But it does put
a new input leaf into the release surface, so the choice belongs to root:

**Should the SPM independence role be delivered to a fresh lineage as a build-stage
input leaf (b), or should the post-hoc enrichment lane be generalised (a)?**

- **Option 1 — (b), the recommendation.** Add the build stage and the
  `DATASET_SOURCE_INPUTS`-driven adapter carve-out. Build P's lane is untouched.
  Cost: a new stage, one adapter exception, moved spec identities. Risk: the
  release's input surface grows by one person column.
- **Option 2 — (a).** Generalise the enrichment to a declared parent, dropping
  pin 5 and reversing the "callers cannot supply their own pins" intent, and turn
  `EXPECTED_COUNTS` into recorded outputs checked against the invariants in §1.
  Cost: comparable code, plus a governance decision about who may declare a
  parent for the calibration-inheritance lane. Risk: the lane's authority is the
  reviewed parent's byte identity, and a fresh base has no inherited calibration
  to carry.
- **Option 3 — both, scoped.** (b) for fresh lineages; keep (a)'s lane pinned to
  Build P for descendants of the certified default.

Nothing in this branch presumes an answer. The preflight check and the release
refusal shipped here are correct under every option: they name the defect and the
remedy without choosing how the remedy is delivered.
