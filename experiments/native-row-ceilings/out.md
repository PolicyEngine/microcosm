# Lane report: the US native build's row-count ceilings

Branch `native-row-ceilings`, from `origin/native-scale-transport` at
`a64f7b733`. Draft PR against `native-scale-transport`, and it stays draft.

The design authority is [`docs/us-native-row-ceilings.md`](../../docs/us-native-row-ceilings.md).
This report is the lane's record: what the census found, what moved and why,
which pins moved, what the tests said verbatim, and what is Max's to decide.

Nothing here is a build, a certification or a release artifact. No gated data was
read. The inputs are one recovered development artifact and one captured public
ACS PUMS archive; every receipt under `experiments/native-row-ceilings/` carries
`"release_eligible": false`.

## 1. The counts, and why they are measurements

Every ceiling below is measured against counts taken from the **catalogues**
inside the recovered 1/1000 pilot preparation artifact
(`_recovered/pilot-runs/native19-required-20260912/run/financial-artifacts/preparation.json`,
sha256 `34b362d85d2f06acd255390976d76343aff45114958ba382edb10ffa3789a8a0`), not
from its rosters.

The distinction decides whether the numbers are measurements or estimates. A
roster at 1/1000 must be scaled to say anything about full source, and scaling a
stratified sample carries error. The catalogues are counts of the whole upstream
ACS and ASEC files and are identical at every fraction. What makes them answer
the full-source question is an identity `roster_census.py` asserts before it
reports anything:

```
ACS occupied_hu       1,348,408
  + institutional_gq     84,422
  + noninstitutional_gq  98,784
  + ASEC households      55,762
  =                   1,587,376  ==  selection.supplied_households
```

"The whole catalogue" and "what a full-source selection supplies" are the same
set, so the catalogue's counts *are* the full-source counts:

| quantity | full source | at 1/10 |
|---|---:|---:|
| ACS households (selectable; excludes 100,355 vacancies) | 1,531,614 | 153,161 |
| ACS persons | 3,422,888 | 342,288 |
| ASEC households | 55,762 | 5,576 |
| ASEC persons | 142,125 | 14,212 |
| **stacked households** | **1,587,376** | 158,737 |
| **stacked persons** | **3,565,013** | 356,501 |
| combined-clone households | 3,174,752 | 317,475 |
| combined-clone persons | 7,130,026 | 713,002 |

**This supersedes the transport lane's figures.** Its §5 quoted ~3,471,000
stacked and ~6,943,000 cloned persons — the 1/1000 artifact's 2.186869 persons
per household extrapolated. The catalogue figures are about 2.7% higher. No
verdict changes; every new constant is derived from these.

One split matters because several bounds are met by one channel rather than the
total: `_normalized_source_copy` normalizes the ACS and ASEC native frames
**separately**, and ACS carries 96.0% of the persons. A per-channel bound is met
by 3,422,888, not 3,565,013.

Receipt: `roster-census.json`, from `roster_census.py`.

## 2. The one argument

> **A bound moves only if a full-source native build meets it. A bound that moves
> becomes four times the measured full-source count of exactly what it counts,
> rounded up to the next whole million.**

The multiple is the transport lane's own — `MAX_ROSTER_BYTES` is 3.9× a
full-source preparation receipt — so both families share one law. The rounding
makes the rule checkable at a glance: divide any moved constant by the count its
comment names and the answer is between 4 and 4.6.

**Byte transports are not this rule's.** The transport lane's answer to a payload
that outgrows its cap is a segmented stream under one explicit total, not a
larger single cap. Applying this rule to one would contradict the neighbouring
argument rather than extend it — and in the clearest case it would not even be
possible: `graph._bounded_json` opens with
`_require(type(limit) is int and 0 < limit <= 64 * 1024**2, "TRANSPORT_LIMIT")`,
so the shared encoder refuses any cap above 64 MiB before it encodes a byte.

**For a bound that looks like a row ceiling, the rule applies one test:** does the
binding site stream, or materialise a per-row payload? A bound in front of a
materialised payload cannot usefully be raised alone, because the payload's own
byte cap refuses first and at a smaller number. §4 shows the three bounds that
test decided.

The rule is also an assertion. `test_us_native_row_ceilings.py` holds the
measured counts and checks each moved ceiling is exactly what the rule produces,
so a later edit that drifts fails a test rather than a build.

## 3. The census

`census.json` is the full table: every `MAX_*` row, household, person, group or
byte bound in `packages/microcosm-build/src/microcosm/build/us_runtime/`
reachable from the 19-node financial graph and the 45-node pilot graph, read at
this head — **146 bounds across five module families** — each with its constant,
its enforcement sites, its refusal code *and exception type*, what it protects,
the count it meets at 1/10 and at full source, and whether it binds.

Every verdict claiming a full-source build meets the bound, or that the bound
guards an encoding width or an upstream file's real size, then went through an
adversarial pass that read the code again and tried to refute it — **42
verdicts**. Many refined a classification or completed an enforcement list.
**None overturned a headline binding call.**

| | |
|---|---:|
| bounds censused | 146 |
| bind at full source, **at the base branch's values** | **19** |
| bind at full source, as `census.json` reports them | 14 |
| bind at 1/10 | **6** |
| verdicts adversarially verified | 42 |
| headline binding calls overturned | 0 |

**Why the two figures differ, stated rather than reconciled quietly.** The census
agents read a moving tree: five of the seven constants §4 moved were lifted while
the census ran, and the agents correctly reported the *post-lift* values at which
those five no longer bind — each of those rows records both numbers and the
commit that changed it. The two that `census.json` still shows as binding,
`current_survey_geography.MAX_HOUSEHOLDS` and `asec_demographic_source._MAX_PERSONS`,
were moved afterwards on its evidence. So: nineteen bind at base, seven moved
here, twelve remain for the transport argument.

**Six bounds bind at 1/10.** That contradicts the transport lane's §5 conclusion
that "a 1/10 build meets no ceiling this lane did not lift". All six are byte
transports:

| bound | value | admits | of source |
|---|---:|---:|---:|
| `acs_person_coverage_authentication.MAX_BODY_BYTES` | 64 MiB | 12,911 ACS persons | **0.38%** |
| `acs_native_coverage_binding.MAX_EVIDENCE_BYTES` | 2 MiB | — | below 1/10 |
| `acs_housing_universe_source.ACS_HU_RECEIPT_MAX_BYTES` | 1 MiB | — | below 1/10 |
| `survey_origin_budget.MAX_PAYLOAD_BYTES` | 64 MiB | 87,838 households | **5.53%** |
| `graph_survey_population.PREPARATION_MAX_BYTES` | 64 MiB | 96,839 households | **6.10%** |
| `graph_current_survey_household_roles.MAX_ARTIFACT_BYTES` | 64 MiB | — | below 1/10 |

The eight that bind only above 1/10: `asec_demographic_source._MAX_PERSONS` and
`current_survey_geography.MAX_HOUSEHOLDS` (both moved here),
`current_child_property_income_source.MAX_ROWS`,
`current_survey_household_roles.MAX_PERSONS`, `graph_survey_age_artifact.MAX_BYTES`,
`graph_survey_calibration.MAX_BYTES` and `.MAX_ROWS`, and
`puf55_survey_recipients.RAW_BYTES_MAX_BYTES` (defined in `microcosm-graph`,
outside this lane's scope), plus `puf_diagnostic_consumer.CURRENT_SURVEY_MAX_BYTES`.

### 3a. `survey_origin_budget.MAX_GROUPS`, established

The transport lane recorded it "not established". It is established, and it
binds. `_initial` builds groups with

```python
instructions = graph.allocation_instructions(
    view.selection_plan, view.receipt["origins"]["households"]
)
_require(0 < len(instructions) <= MAX_GROUPS, "GROUP_COUNT_BOUND")
```

and `allocation_instructions` opens with
`_require(len(household_origins) == len(plan.selected), "ORIGIN_COUNT")`. One
instruction per selected household, so the group count *is* the selected
household count: **1,587,376 at full source against a 1,000,000 bound.**

Where it runs, stated precisely. **No node in the nineteen-node graph executes
`freeze_survey_origin_budget`**, which is where `GROUP_COUNT_BOUND` is checked —
the recovered run's `graph.json` lists all nineteen and none is a budget node,
and the completion host has none either. `graph_atomic_survey_financial` does
call into the module, but only for `_config_payload` and the `_live()` producer
seal, neither of which reaches `_initial`. `survey_age_calibration` and
`graph_survey_budget` are what execute it, over the same full-source selection,
so the count and the verdict stand.

## 4. What moved

| constant | was | now | counts | full source | × |
|---|---:|---:|---|---:|---:|
| `acs_pums.MAX_EXACT_HOUSEHOLDS` | 1,000,000 | **7,000,000** | exact ACS household keys | 1,531,614 | 4.57 |
| `acs_pums.MAX_EXACT_PERSON_ROWS` | 1,000,000 | **14,000,000** | `NP` over selected ACS households | 3,422,888 | 4.09 |
| `acs_person_coverage_columns.MAX_SELECTED_ROWS` | 1,000,000 | **14,000,000** | requested ACS person keys | 3,422,888 | 4.09 |
| `survey_observed_age.MAX_ROWS` | 2,000,000 | **14,000,000** | one channel's person rows | 3,422,888 | 4.09 |
| `survey_origin_budget.MAX_GROUPS` | 1,000,000 | **7,000,000** | allocation instructions = selected households | 1,587,376 | 4.41 |
| `current_survey_geography.MAX_HOUSEHOLDS` | 524,288 | **7,000,000** | selected households | 1,587,376 | 4.41 |
| `asec_demographic_source._MAX_PERSONS` | 600,000 | **14,000,000** | retained ACS persons (the larger of its two rosters) | 3,422,888 | 4.09 |

Every refusal keeps its code, its exception type and its expression. Only the
number moves.

| constant | refusal | raises |
|---|---|---|
| `MAX_EXACT_HOUSEHOLDS` | `"ACS exact selection requires bounded unique raw native keys."` | `ValueError` |
| `MAX_EXACT_PERSON_ROWS` | `"ACS selected complete roster exceeds native person budget."` | `ValueError` |
| `MAX_SELECTED_ROWS` | `"ACS coverage selected person count is outside the bound"`; `"SELECTED_ROWS"`, `"NATIVE_ROWS"`, `"NATIVE_ROW_BUDGET"` downstream | `ValueError`, `ACSCoverageAuthenticationError`, `ACSNativeCoverageBindingError` |
| `survey_observed_age.MAX_ROWS` | `"SURVEY_OBSERVED_AGE_ROW_BOUND"` | `ValueError` |
| `MAX_GROUPS` | `"GROUP_COUNT_BOUND"` | `SurveyOriginBudgetError` |
| `current_survey_geography.MAX_HOUSEHOLDS` | `"CURRENT_SURVEY_GEOGRAPHY_HOUSEHOLD_COUNT"`, `…_PROJECTION_STORAGE` | `ValueError` |
| `asec_demographic_source._MAX_PERSONS` | `"ACS_ROWS"`, `"MEMBERSHIP_ROWS"`, `"CLASSIFY_ROWS"`, `"DEMOGRAPHIC_ROWS"` | `ValueError` |

The last two were not in the brief and were moved only after reading what stood
behind them:

- **`current_survey_geography.MAX_HOUSEHOLDS` bound hardest of the seven** —
  524,288 against 1,587,376, refusing at 33% of source. It reads as a byte budget
  (`64 * 1024**2 // 128`) but **streams**: `_projection_digest` feeds one bounded
  row encoding at a time into a `hashlib.sha256`, the module materialises nothing
  per household, and its only bytes are a 64 KiB summary receipt already bounded
  by `MAX_RECEIPT_BYTES`. The borrowed 64 MiB never described anything here.
- **`asec_demographic_source._MAX_PERSONS` is one constant over two rosters**, so
  it takes the larger. It is not a fixed-width encoding: `rows` is a JSON header
  integer, the only `struct.pack` is `"<I"` over the *header* length which
  `_HEADER_MAX` bounds separately, and the body budget is derived and checked
  against actual bytes rather than capped. And it could not have been a claim
  about the ACS file's size — at 600,000 it sat **5.7× below** the genuine
  3,422,888-row file, and a bound that would refuse the real file is not an
  assertion about it. `ACS_SOURCE_ROW_SHAPE` and `ACS_CAPTURE_CHANGED` make that
  claim exactly, and `DEMOGRAPHIC_COHORT_ROWS` makes the ASEC one; neither
  weakens.

Two bounds that look the same failed the streaming test and were left alone:
`current_survey_household_roles.MAX_PERSONS` sits behind
`table.reset_index().to_json(orient="table").encode()` under a 64 MiB artifact
cap, and `current_child_property_income_source.MAX_ROWS` behind one `json.dumps`
under a 64 MiB projection cap.

## 5. What did not move, and why

- **An upstream file's real size.** `acs_person_coverage_columns.MAX_ROWS` and
  `acs_native_coverage_binding.MAX_SOURCE_ROWS`, both 6,000,000, assert that a
  genuine ACS archive is near its real 3,422,888 records. Raising either trades a
  real check for nothing: a selection cannot exceed its source, and the selected
  roster is bounded separately.
- **A fixed-width encoding.** The `PAYLOAD_MAX_BYTES = len(MAGIC) + 4 +
  HEADER_MAX_BYTES + ROW_BYTES * MAX_HOUSEHOLDS + 32` family in
  `asec_housing_status`, `asec_housing_universe`, `asec_income_observations`,
  `_asec_current_money_codec`, `asec_current_money_selection` and
  `graph_asec_income` computes a byte budget *from* a row ceiling, so moving the
  row ceiling moves the wire format. `survey_observed_age.MAX_EXACT_FLOAT64_INTEGER`
  = 2**53 bounds a value, not a row count; its own docstring calls it "a
  representation limit, not a scientifically permitted maximum age".
- **A quantity no fraction grows.** `asec_current_money.MAX_PERSONS` (1,000,000)
  and `MAX_HOUSEHOLDS` (400,000) count the ASEC source scope's own `person_rows`
  and `household_rows`, measured at 142,125 in 55,762. **The transport lane's §5
  listed `asec_current_money.MAX_PERSONS` among the bounds a full-source build
  meets. It does not, and this lane corrects that.**
- **Why a bound stays a bound rather than being deleted.** Each stands between a
  malformed or runaway input and an unbounded allocation, and the refusal makes
  the failure legible and early rather than an OOM kill deep in a multi-hour
  build with no statement of which roster was wrong. A ceiling at 4× the real
  thing still refuses the shape a real mistake takes: a fraction argument off by
  a decimal place, a selection that forgot to filter, keys built from the wrong
  catalogue.

## 6. Three ceilings that refuse before anything this lane moved

All three are byte transports. None is this lane's to move. All three are
measured and pinned in `test_us_native_row_ceilings.py`, so the next reader meets
them in a test rather than in a build.

**1. `acs_person_coverage_authentication.MAX_BODY_BYTES` is the tightest ceiling
on the path, and it is not close.** It charges every *selected* row
`6 * len(raw) + 1024` bytes before the reader allocates its DataFrame, refusing
`SELECTED_BODY_BUDGET`. The only variable is the record's own length, so the
admitted count is measurable rather than estimated — `selected_body_budget.py`
streams 200,000 real records of the pilot's captured public ACS PUMS archive:

| role | record bytes (mean / max) | charge per row | admits | of source | at full source |
|---|---:|---:|---:|---:|---:|
| person | 695.57 / 852 | 5,197 | **12,911** | **0.38%** | 265× over |
| household | 589.12 / 756 | 4,559 | 14,720 | 0.96% | 104× over |

That is below 1/100. It is also why lifting
`acs_person_coverage_columns.MAX_SELECTED_ROWS` is **necessary and not
sufficient**: this refuses 265× earlier on the same path.

**2. The preparation-receipt ceiling the transport lane lifted is still enforced
one module downstream.** That lane raised `survey_population_preparation`'s total
to `MAX_ROSTER_BYTES` = 64 × 64 MiB and reported the receipt ceiling moved from
96,860 households to 6,206,000. `_roster_payload` still returns one joined
payload — it must, because `KernelResult.artifacts` is a mapping of `bytes` — and
`graph_survey_population._checked_preparation` checks those same bytes against
`PREPARATION_MAX_BYTES`, **still 64 MiB**, refusing `PREPARATION_BYTES` at `:305`
and `:309`, and again in the allocation kernel at `:577`/`:582`. Six call sites
reach it, including `graph_atomic_survey_population:172`.

From the transport lane's own committed `ceiling-receipt.json`: a 1/10 roster
measured 109,804,304 bytes and was recorded **`accepted`** by the producer —
1.64× this cap — and a full-source roster measured 1,099,892,722 bytes, of which
this cap admits **96,839 households**. To within rounding, exactly the 96,860
that lane reported as the ceiling it had lifted. Receipt: `consumer-gap.json`.

**3. `survey_origin_budget.MAX_PAYLOAD_BYTES` admits 87,838 households, 5.53% of
source** — below 1/10 and below the 96,860 the transport lane lifted. Measured
through the module's own `_reference` and `_json` at full-source household-id
widths: 764 bytes per group including the two `household_ids` and two
`group_indices` the header carries per clone pair; 1.13 GiB at full source, 18.07×
the cap. Receipt: `origin-budget-size.json`.

It cannot be raised in that module at all: `_json` is
`graph._bounded_json(value, MAX_PAYLOAD_BYTES)`, and `_bounded_json` refuses any
limit above 64 MiB before encoding a byte. A larger number would refuse the
module, not loosen it.

So **lifting `MAX_GROUPS` is necessary and not sufficient**: `GROUP_COUNT_BOUND`
is checked in `_initial` before the payload is built, so at full source it fired
first; with it lifted the refusal moves to `TRANSPORT_LIMIT` in the same module.
That is the correct outcome of one argument applied to one class of bound.

## 7. Pins

Re-derived through their generators, never hand-edited. `repin.py` reports all of
this in one run; both generators are idempotent.

| pin | old | new | generator |
|---|---|---|---|
| `graph_implementation_inventory.json` → `contracts` → `survey_population_preparation.py` → `resource_accesses_sha256` | `ccfed1c1acff5a1c50b538424dc129930f69c374d9841e233b3fb593a440c3aa` | `0071f934801d15c14e4be66af12ff73daee5f65c89f61d655a23e462afdcd061` | `graph_implementation._dependency_contract(payload, name, _covered_imports(name, inventory))`, via `regenerate_inventory_contract.py --write` |
| `acs_native_coverage_binding._ACCEPTED["acs_pums.py"]` | `6ecf79f0dfb0c0bc0ad0af6be5fa65bd8c2d1e1968009402e4c8dee347de70dc` | `e79a2a4ecbc81e52b361918e7f391725a336274c74041387d71195cca46cb83c` | `sha256(<module>.read_bytes())`, the same call the check makes, via `regenerate_accepted_pin.py --write` |
| the other 123 inventory contracts | — | **unchanged** | same generator, over all of them |
| `_ACCEPTED["acs_inputs.py"]`, `["acs_housing_universe_source.py"]`, `["acs_person_coverage_authentication.py"]` | — | **unchanged** | same |
| all ten `implementation_manifest(stage)` | — | **built** | `graph_implementation.implementation_manifest(stage)` |

**The first row is not this lane's, and it is a blocker on the base branch.** On
`native-scale-transport`, `b6081efcb` ("Hash a spill segment that was already
there") added `path.read_bytes()` inside `_spill_roster`. `read_bytes` is in
`graph_implementation._RESOURCE_CALLS`, so it moves that module's
`resource_accesses_sha256` — and the pin was not regenerated.
`implementation_manifest()` therefore raised `"Unclassified US
dependency/resource contract"` for **every stage containing that module**,
including `authenticated_survey_population_v1`, the stage the nineteen-node path
runs.

Proven pre-existing two ways. The contract was recomputed from the blobs of
`origin/native-scale-transport`, `a64f7b733`, `b6081efcb` and its four
predecessors: it matches at `a51957c5c` and earlier and mismatches from
`b6081efcb` on. And the affected tests were run against a base worktree at
`a64f7b733` — **7 failed there with that exact `ValueError`, and all 7 pass
here.** This branch does not touch that file.

The transport lane's report §4 was right that `Path.write_bytes` is invisible to
`resource_accesses_sha256` by construction. This is the converse: a later commit
on the same branch added a *read*, which is visible.

What moves and is not a committed pin: each US stage's `implementation_hash` is
over its whole module roster, so editing any inventoried module moves it and with
it every node key and store address. #935 recorded that this holds for any change
to these files including a comment.

## 8. Gates, verbatim

All at `f40294e91`'s tree, with the tree untouched for the duration of the
pytest run. Every command's own last line, quoted rather than summarised.

**The touched test files, as CI runs them — explicit flat file list, no `-k`:**

```
$ .venv/bin/python -m pytest \
    packages/microcosm-build/tests/test_us_native_row_ceilings.py \
    packages/microcosm-build/tests/test_us_acs_pums.py \
    packages/microcosm-build/tests/test_us_acs_person_coverage_columns.py \
    packages/microcosm-build/tests/test_us_survey_observed_age.py \
    packages/microcosm-build/tests/test_us_current_survey_geography.py \
    packages/microcosm-build/tests/test_us_asec_demographic_source.py \
    packages/microcosm-build/tests/test_us_survey_origin_budget.py -rf

399 passed in 481.27s (0:08:01)
```

**The CI group gates:**

```
$ .venv/bin/python tools/ci_test_groups.py --verify
verification=ok
exit=0

$ python3 -I -B -S packages/microcosm-build/tests/test_ci_test_groups.py
Ran 15 tests in 0.396s

OK

$ .venv/bin/python tools/spec_engine_coverage.py --check
spec-engine coverage: 42156/42156 configuration fields; 41/41 inventory checks
exit=0
```

`test_us_native_row_ceilings.py` lands in `rest` shard 3/6 (fast), `us-not`
shard 1/1 (engine) and `wheels`, and never under `[defaulted]`. The other six
touched files keep their existing groups: `us-am` for the four `a`–`m` files,
`us-qs` for the two `s` files.

**Lint and format, over every `.py` this branch touches:**

```
$ git diff --name-only origin/native-scale-transport...HEAD | grep '\.py$' \
    | xargs .venv/bin/python -m ruff check
All checks passed!
exit=0

$ git diff --name-only origin/native-scale-transport...HEAD | grep '\.py$' \
    | xargs .venv/bin/python -m ruff format --check
21 files already formatted
exit=0
```

**The pin generators, re-run at this head:**

```
$ .venv/bin/python experiments/native-row-ceilings/repin.py .../repin.json
inventory contracts checked: 124; moved: 0
acs_native_coverage_binding._ACCEPTED:
  acs_pums.py: unchanged
  acs_inputs.py: unchanged
  acs_housing_universe_source.py: unchanged
  acs_person_coverage_authentication.py: unchanged
stage manifests built: 10

$ .venv/bin/python experiments/native-row-ceilings/regenerate_inventory_contract.py
no contract moved

$ .venv/bin/python experiments/native-row-ceilings/regenerate_accepted_pin.py
no pin moved
```

**One earlier run reported failures, and it was not a valid measurement.** A
full-file run of `test_us_survey_origin_budget.py` reported `8 failed, 44
passed` while this session was editing the working tree underneath it. The same
tests pass at a fixed commit — the subset run returned `8 passed, 44 deselected
in 196.03s`, and the run above is green over the whole file. It is recorded here
because it was run and reported, not because it measured anything.

**Against the base branch, for the inherited pin.** Seven of those tests were run
against a worktree at `a64f7b733` with `PYTHONPATH` pointing at its own packages:

```
7 failed, 44 deselected in 59.22s
E   ValueError: Unclassified US dependency/resource contract:
    microcosm.build/us_runtime/survey_population_preparation.py.
```

All seven pass on this branch. That is the evidence in §7 that the break is the
base's and not this lane's.

## 9. For Max

**1. The inherited re-pin — keep it here, or move it to #945?**
`b6081efcb` on `native-scale-transport` added `path.read_bytes()` to
`_spill_roster` without regenerating the pin, and `implementation_manifest()`
raises for the stage the nineteen-node path runs. Seven tests fail on a base
worktree with that exact error; all seven pass here.

- **(a) Keep it in this PR.** One generated line; it makes the base functional
  and lets this lane build its own stage manifests. Costs: this PR touches a
  file that lane owns.
- **(b) Move it to #945 and rebase this branch.** Cleaner ownership. Costs: this
  branch cannot build a stage manifest until #945 carries the fix, so its pin
  evidence is unverifiable in the meantime.
- **(c) Keep it here *and* tell #945's owner**, so the fix is not silently
  inherited and re-derived twice.

I shipped (a) and would pick (c). I did not pick (b) because required step 4 of
this lane's brief is to re-derive every pin through its generator, and the
generator raises on an unfixed base.

**2. The twelve byte transports that still bind — one lane or several?**
Six of them bind at **1/10**, and `acs_person_coverage_authentication.MAX_BODY_BYTES`
binds at **0.38% of source**. They are all the same shape and the argument for
them already exists, written by the transport lane.

- **(a) One follow-up lane, one argument**, carrying the segmented transport into
  all of them — the same reasoning that produced this lane.
- **(b) One lane per module, in binding order**: the ACS body budget (0.38%), the
  preparation consumer (6.10%), the origin budget (5.53%), the roles artifact,
  the calibration pair, the age artifact.
- **(c) Only the ones on the nineteen-node path now**, leaving the calibration and
  age-artifact ones to whoever meets them.

My reading is (a), for the reason you already gave on this family. But the ACS
body budget is different in kind from the rest — it refuses at 1/265, so it
gates *any* run above the 1/1000 pilot, and it may deserve to go first whatever
the shape of the rest.

**3. Seven moved, not the four the brief named. Keep them together?**
The brief named four; `MAX_GROUPS` it asked me to establish, and establishing it
showed it binds. The census then found two more that the rule decides with no
judgment left — `current_survey_geography.MAX_HOUSEHOLDS`, which bound hardest of
all seven at 33% of source, and `asec_demographic_source._MAX_PERSONS`.

- **(a) Keep all seven in this PR.** One rule applied wherever it decides.
- **(b) Split the two the brief did not name into their own PR.**

I shipped (a): splitting them is the "one build at a time" you ruled out, and
both were moved on the same read-the-code test as the rest, recorded in §4.

**4. The multiple — and one place a tighter rule exists.**
I used four times the measured full-source count, rounded up to the next whole
million, because that is the transport lane's own headroom (3.9×) and it keeps
one law over both families.

- **(a) Keep 4×.**
- **(b) A larger multiple** — 10× would be about three decades of ACS growth
  rather than one, at no runtime cost, since these are ceilings and not
  allocations.
- **(c) Where a containing bound already exists, use it instead.**
  `MAX_SELECTED_ROWS` bounds requested person keys and `MAX_ROWS` bounds the
  archive those keys are drawn from, so `MAX_SELECTED_ROWS = MAX_ROWS` would be
  tighter than 14,000,000 *and* provable by containment rather than chosen. It
  would make the rule two rules, which is why I did not.

I shipped (a). (c) is the only one I think is genuinely arguable, and only for
that one pair.

## 10. The pull request

| | |
|---|---|
| PR | **[PolicyEngine/microcosm#949](https://github.com/PolicyEngine/microcosm/pull/949)** — draft, and it stays draft |
| Title | Lift the seven row-count ceilings a full-source native build meets, under one rule |
| Base | `native-scale-transport` (PR #945's branch), at `a64f7b733` |
| Head | what `git rev-parse native-row-ceilings` returns; this table does not quote a sha it cannot have written |
| Mergeable | `MERGEABLE`, verified at the head this report was written against |
| `packages/microcosm-graph` hunks | **zero** |
| CI | **does not run on this PR by design.** `.github/workflows/test.yml` triggers on `pull_request: branches: [main]`, so only a PR targeting `main` reaches it, and `gh pr checks 949` reports none. §8 is the only gate this branch has, and it was run locally. |

Files, against the base:

| file | what |
|---|---|
| `.../us_runtime/acs_pums.py` | the two exact-selection ceilings |
| `.../us_runtime/acs_person_coverage_columns.py` | the requested-roster ceiling; `MAX_ROWS` deliberately unchanged, with the reason in a comment |
| `.../us_runtime/survey_observed_age.py` | the per-channel row ceiling |
| `.../us_runtime/survey_origin_budget.py` | `MAX_GROUPS`; `MAX_PAYLOAD_BYTES` deliberately unchanged, with the reason in a comment |
| `.../us_runtime/current_survey_geography.py` | the household ceiling, and why its byte-derived form never described anything here |
| `.../us_runtime/asec_demographic_source.py` | the person ceiling over its two rosters |
| `.../us_runtime/acs_native_coverage_binding.py` | the one moved pin, generated |
| `.../us_runtime/graph_implementation_inventory.json` | the **inherited** re-pin, generated |
| `packages/microcosm-build/tests/test_us_native_row_ceilings.py` | the rule as an executable table, the bounds that must not move, and the three consumer ceilings pinned |
| six existing `test_us_*.py` | one boundary test per moved bound |
| `docs/us-native-row-ceilings.md` | the design authority |
| `experiments/native-row-ceilings/` | five measurement tools, three pin tools, and their receipts |
| `changelog.d/native-row-ceilings.changed.md` | the towncrier fragment |
| `pyproject.toml` | four per-file `E402` ignores, for tools that must set `sys.path` before importing what they measure |
| `PROGRESS-native-row-ceilings.md` | the lane journal |

## 11. What a reader should not take from this report

- **Nothing here is a build, a certification or a release artifact.** Every
  receipt carries `"release_eligible": false` and a scope line. No gated data was
  read: the inputs are one recovered development artifact and one captured
  **public** ACS PUMS archive, both read by path, neither written, moved or
  linked.
- **The census agents read a moving tree.** Five constants were lifted while the
  census ran; those rows record both values and the commit that changed them, and
  §3 reconciles the two totals rather than quoting one.
- **`census.json`'s per-row prose is the agents', checked but not rewritten.**
  Every verdict that claimed a bound binds at full source, or guards an encoding
  width or an upstream file's size, was adversarially re-read — 42 of them — and
  several classifications were corrected in that pass. The rows themselves were
  not edited afterwards, so a row may still carry a superseded classification
  beside its verified `binds_at_full_source`. The verified field is the one this
  report relies on, and every number this report or the design note states was
  re-read at this head before being written down.
- **The roles projection figure is on an invented table.** It is faithfully
  shaped — the module's own declared columns and dtypes, its own encoder — and the
  per-row cost is a difference between two row counts so the schema preamble
  cancels. The 320.08 bytes is not a measurement of a real roles artifact.
- **The 12,911 and 87,838 figures are of a bound, not of a run.** They say what
  the ceiling admits, computed from a measured per-row cost. No build was run at
  any fraction in this lane.
