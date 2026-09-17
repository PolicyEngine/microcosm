# US native build: the row-count ceilings, and the rule their new values follow

The [transport lane](us-native-scale-transport.md) lifted the 64 MiB
whole-roster receipt ceilings and, in its §5, recorded a second family it did
not touch: row-count bounds that a full-source native US build meets. This note
is that family's argument.

It is one rule, not five decisions. Max decided (2026-09-17) that these are
lifted as one change with one argument rather than one build at a time, and a
rule is the only thing that makes the sixth case decide itself.

## 1. The rule

> **A bound moves only if a full-source native build meets it. A bound that moves
> becomes four times the measured full-source count of exactly what it counts,
> rounded up to the next whole million.**

Three clauses, each load-bearing.

**"only if a full-source build meets it."** Most of the bounds censused in §3 do
not bind, and most of those never will, because what they count is an upstream
file's own size rather than a roster this build chooses. A rule that raised
everything uniformly would weaken real checks for no gain. This one leaves them
alone by construction, and §4 says why each must stay.

**"of exactly what it counts."** The commonest way to get one of these wrong is
to bound the wrong roster. The 19-node path carries at least five distinct
quantities that differ by more than an order of magnitude — the upstream ACS
file's rows, the selected ACS roster, the selected ASEC roster, the stacked
roster, and the combined clone — and a ceiling raised against the wrong one is
either dead or still binding. §2 measures each, and every moved constant carries
a comment naming the one it counts.

**"four times … rounded up to the next whole million."** The multiple is the
transport lane's: `MAX_ROSTER_BYTES` is `64 × MAX_SEGMENT_BYTES` = 4 GiB, which
that lane measured at 3.9× a full-source preparation receipt. Adopting the same
headroom keeps one law across both families. The rounding makes the constants
legible and the rule checkable at a glance: divide any moved constant by the
count its comment names and the answer is between 4 and 4.6.

The rule is not only written here. `test_us_native_row_ceilings.py` holds the
measured counts and asserts that each moved ceiling is exactly what the rule
produces from its count, so a later edit that drifts fails a test rather than a
build.

### 1a. What the rule is not for

**Byte transports take the transport lane's argument, not this one.** That
lane's answer to a payload that outgrows its cap is a segmented stream under one
explicit total, not a larger single cap, and it proved the segmented bytes
identical to the unsegmented ones. Raising such a cap here would contradict the
neighbouring argument rather than extend it. §5 names the byte ceilings that
bind and leaves them to it — including one that binds harder than anything this
lane moved.

## 2. The counts, measured rather than extrapolated

Every number below comes from the recovered 1/1000 pilot preparation artifact
(`_recovered/pilot-runs/native19-required-20260912/run/financial-artifacts/preparation.json`,
sha256 `34b362d85d2f06acd255390976d76343aff45114958ba382edb10ffa3789a8a0`), and
specifically from its **catalogues** rather than its rosters.

That distinction is the point. A roster at 1/1000 has to be scaled to say
anything about full source, and scaling a stratified sample introduces error.
The catalogues do not: they are counts of the whole upstream ACS and ASEC files,
and they are identical in a 1/1000 artifact and a full-source one. What makes
them answer the full-source question is this identity, which
`experiments/native-row-ceilings/roster_census.py` asserts before reporting
anything:

```
ACS occupied_hu   1,348,408
  + institutional_gq  84,422
  + noninstitutional_gq 98,784
  + ASEC households    55,762
  = 1,587,376  ==  selection.supplied_households
```

"The whole catalogue" and "what a full-source selection supplies" are therefore
the same set, and the catalogue's counts *are* the full-source counts:

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

The transport lane's §5 quoted ~3,471,000 stacked and ~6,943,000 cloned persons.
Those were the 1/1000 artifact's 2.186869 persons per household extrapolated,
and the catalogue figures supersede them — about 2.7% higher, which changes no
verdict but is the number each new constant is derived from.

One further split matters, because two of the moved bounds are met by one
channel rather than the total: **`_normalized_source_copy` normalizes the ACS
and ASEC native frames separately**, and the ACS channel carries 96.0% of the
persons. A bound met per channel is met by 3,422,888, not by 3,565,013.

## 3. The moved ceilings

| constant | was | now | counts | full source | × |
|---|---:|---:|---|---:|---:|
| `acs_pums.MAX_EXACT_HOUSEHOLDS` | 1,000,000 | **7,000,000** | exact ACS household keys in one snapshot | 1,531,614 | 4.57 |
| `acs_pums.MAX_EXACT_PERSON_ROWS` | 1,000,000 | **14,000,000** | `NP` summed over the selected ACS households | 3,422,888 | 4.09 |
| `acs_person_coverage_columns.MAX_SELECTED_ROWS` | 1,000,000 | **14,000,000** | requested ACS person keys | 3,422,888 | 4.09 |
| `survey_observed_age.MAX_ROWS` | 2,000,000 | **14,000,000** | one channel's person rows | 3,422,888 | 4.09 |
| `survey_origin_budget.MAX_GROUPS` | 1,000,000 | **7,000,000** | allocation instructions = selected households | 1,587,376 | 4.41 |
| `current_survey_geography.MAX_HOUSEHOLDS` | 524,288 | **7,000,000** | selected households | 1,587,376 | 4.41 |
| `asec_demographic_source._MAX_PERSONS` | 600,000 | **14,000,000** | retained ACS persons (the larger of its two rosters) | 3,422,888 | 4.09 |

Each refusal keeps its code, its exception type and its expression; only the
number moves.

| constant | refusal | raises |
|---|---|---|
| `MAX_EXACT_HOUSEHOLDS` | `"ACS exact selection requires bounded unique raw native keys."` | `ValueError` |
| `MAX_EXACT_PERSON_ROWS` | `"ACS selected complete roster exceeds native person budget."` | `ValueError` |
| `MAX_SELECTED_ROWS` | `"ACS coverage selected person count is outside the bound"` | `ValueError` |
| `survey_observed_age.MAX_ROWS` | `"SURVEY_OBSERVED_AGE_ROW_BOUND"` | `ValueError` |
| `MAX_GROUPS` | `"GROUP_COUNT_BOUND"` | `SurveyOriginBudgetError` |
| `current_survey_geography.MAX_HOUSEHOLDS` | `"CURRENT_SURVEY_GEOGRAPHY_HOUSEHOLD_COUNT"`, `…_PROJECTION_STORAGE` | `ValueError` |
| `asec_demographic_source._MAX_PERSONS` | `"ACS_ROWS"`, `"MEMBERSHIP_ROWS"`, `"CLASSIFY_ROWS"`, `"DEMOGRAPHIC_ROWS"` | `ValueError` |

### 3a. `survey_origin_budget.MAX_GROUPS`, which the transport lane left open

Its §5 recorded this one as "not established — the group count was not derived
here". It is established now, and it binds. `_initial` builds its groups with

```python
instructions = graph.allocation_instructions(
    view.selection_plan, view.receipt["origins"]["households"]
)
_require(0 < len(instructions) <= MAX_GROUPS, "GROUP_COUNT_BOUND")
```

and `allocation_instructions` opens with
`_require(len(household_origins) == len(plan.selected), "ORIGIN_COUNT")`. One
instruction per selected household, so the group count *is* the selected
household count: 1,587,376 at full source against a 1,000,000 bound.

Two things about where it runs, stated rather than implied. It is reachable by
import from `graph_atomic_survey_financial`, but **no node in the nineteen-node
graph executes it** — the recovered run's `graph.json` lists all nineteen and
none is a budget node — and none in the completion host does either.
`survey_age_calibration` and `graph_survey_budget` execute it, over the same
full-source selection, so the count and the verdict are unchanged.

## 4. Why a bound stays a bound, and why some must not move at all

**Deleting a bound is not the cheap version of raising it.** Each of these
stands between a malformed or runaway input and an unbounded allocation, and the
refusal is what makes the failure legible and early rather than an OOM kill deep
in a multi-hour build with no statement of which roster was wrong. A ceiling at
4× the real thing still refuses a 10× runaway, which is the shape a real mistake
takes: a fraction argument off by a decimal place, a selection that forgot to
filter, a key list built from the wrong catalogue.

Three classes must not move at all, and this note names them so a later reader
does not mistake a low number for an oversight.

**An upstream file's real size.** `acs_person_coverage_columns.MAX_ROWS` and
`acs_native_coverage_binding.MAX_SOURCE_ROWS`, both 6,000,000, bound how many
records the reader will stream out of the ACS archive. The 2024 ACS person file
holds 3,422,888 of them. That bound is a structural assertion about a file this
build does not produce and cannot check any other way, and the roster it feeds
is separately bounded by `MAX_SELECTED_ROWS`. Raising it would trade a real
check for nothing: a selection cannot exceed the source it is drawn from.

**A fixed-width encoding.** The `PAYLOAD_MAX_BYTES = len(MAGIC) + 4 +
HEADER_MAX_BYTES + ROW_BYTES * MAX_HOUSEHOLDS + 32` family in
`asec_housing_status`, `asec_housing_universe` and `asec_income_observations`
computes a byte budget *from* a row ceiling, so moving the row ceiling moves the
wire format a reader depends on. `survey_observed_age.MAX_EXACT_FLOAT64_INTEGER`
= 2**53 is the same kind of thing one level down: the module's own docstring
calls it "a representation limit, not a scientifically permitted maximum age",
and it bounds an age's value, not how many rows carry one.

**A quantity no selection fraction grows.** The ASEC codec's
`asec_current_money.MAX_PERSONS` (1,000,000) and `MAX_HOUSEHOLDS` (400,000)
count the ASEC source scope's own `person_rows` and `household_rows`. The
catalogue measures those at 142,125 persons in 55,762 households, and a
full-source survey selection does not change them. **The transport lane's §5
listed `asec_current_money.MAX_PERSONS` among the bounds a full-source build
meets; it does not, and this note corrects that.**

## 5. What still binds, and whose argument it is

The census in §6 found fourteen bounds a full-source build meets. Seven are the
row counts §3 moved. The rest belong to the transport lane's argument — a
segmented stream under one explicit total — not to this one.

### 5a. The test the rule applies

Three of the remaining bounds *look* like row ceilings, and one of them was
moved here after that test was applied to it. The test is a single question,
answered from code rather than from the constant's name:

> **Does the binding site materialise a per-row payload, or does it stream?**

A bound in front of a materialised payload cannot be usefully raised on its own,
because the payload's own byte cap refuses first and at a smaller number. A
bound in front of a streaming digest has nothing behind it, and the rule applies.

| bound | value | what is behind the binding site | verdict |
|---|---:|---|---|
| `current_survey_geography.MAX_HOUSEHOLDS` | 524,288 | `_projection_digest` streams one `_encode(values, maximum=4096)` at a time into a `hashlib.sha256`; the module's only bytes are a 64 KiB summary receipt | **streams — moved in §3** |
| `current_survey_household_roles.MAX_PERSONS` | 2,097,152 | `_projection_bytes` is `table.reset_index().to_json(orient="table").encode()` — the whole per-person table in one string — under `graph_current_survey_household_roles.MAX_ARTIFACT_BYTES` = 64 MiB | materialises — transport's |
| `current_child_property_income_source.MAX_ROWS` | 600,000 | `_json` is one `json.dumps(value)` under `MAX_PROJECTION_BYTES` = 64 MiB | materialises — transport's |

The middle row is the clearest case for why the test matters: at roughly 200
bytes of JSON per person, that 64 MiB cap admits a few hundred thousand persons,
so raising the 2,097,152 row bound would move nothing at all.

A fourth bound needed the same test plus one more question, and passed both.
`asec_demographic_source._MAX_PERSONS` (600,000) is enforced at five sites that
count **two different quantities**: a fixed ASEC three-cohort roster of 432,523
rows, and the retained ACS person roster, which a full-source build grows to
3,422,888. One constant over two rosters takes the larger, so the rule produces
14,000,000 from the ACS arm. Three things had to hold before it could move, and
each was read rather than assumed:

- **It is not a fixed-width encoding.** `rows` lives as a JSON integer in the
  header; the module's only `struct.pack` is `"<I"` over the *header length*,
  which `_HEADER_MAX` bounds separately. The body budget is derived and checked
  against the actual bytes (`len(self._body) == rows * len(COLUMNS) * 8`), never
  capped, so nothing sizes an allocation or an offset from `_MAX_PERSONS`.
- **It is not an assertion about the ACS file's size** — it could not be. At
  600,000 it sat **5.7× below** the genuine 3,422,888-row ACS person file, and a
  bound that would refuse the real file cannot be a structural claim about it.
  That claim is made exactly elsewhere, by `ACS_SOURCE_ROW_SHAPE` and
  `ACS_CAPTURE_CHANGED`; the ASEC arm's rows are asserted exactly by
  `DEMOGRAPHIC_COHORT_ROWS`. Neither check weakens when the shared ceiling moves.
- **No per-row payload sits behind the binding site.** `:1016` takes two numpy
  arrays.

### 5b. Three the owner should see

The census found **six** bounds that a **1/10** build meets, all of them byte
transports. That contradicts the transport lane's §5 conclusion that "a 1/10
build meets no ceiling this lane did not lift", and two of them matter enough to
be measured rather than listed.

**The ACS coverage authentication body budget is the tightest ceiling on the
whole path, and it is not close.**
`acs_person_coverage_authentication` charges every *selected* row
`6 * len(raw) + 1024` bytes against `MAX_BODY_BYTES` (64 MiB) before the reader
allocates its DataFrame, refusing `SELECTED_BODY_BUDGET`. The only variable is
the record's own length, so the admitted count is measurable: over 200,000 real
records of the pilot's captured public ACS PUMS archive, a person record
averages 695.57 bytes, so the charge is 5,197 bytes and the budget admits
**12,911 selected persons — 0.38% of source**, 265× under at full source. The
household role admits 14,720, 0.96%.

That is below 1/100. It is a byte transport, so it is the transport argument's,
and it is why lifting `acs_person_coverage_columns.MAX_SELECTED_ROWS` in §3 is
**necessary and not sufficient**: this refuses 265× earlier on the same path.

**The preparation-receipt ceiling the transport lane lifted is still enforced one
module downstream.** That lane raised `survey_population_preparation`'s total to
`MAX_ROSTER_BYTES` = 64 × 64 MiB and reported the receipt ceiling moved from
96,860 households to 6,206,000. `_roster_payload` still returns one joined
payload — it must, because `KernelResult.artifacts` is a mapping of `bytes` —
and `graph_survey_population._checked_preparation` checks those same bytes
against `PREPARATION_MAX_BYTES`, **still 64 MiB**, refusing `PREPARATION_BYTES`
(`:305`, `:309`, and again in the allocation kernel at `:577`/`:582`), reached
from six call sites including `graph_atomic_survey_population:172`.

From the transport lane's own committed `ceiling-receipt.json`: a 1/10 roster
measured 109,804,304 bytes and was recorded `accepted` by the producer — 1.64×
this cap — and a full-source roster measured 1,099,892,722 bytes, of which this
cap admits **96,839 households**. That is, to within rounding, exactly the 96,860
the transport lane reported as the ceiling it had lifted.

Both are pinned in `test_us_native_row_ceilings.py`, so the next reader meets
them in a test rather than in a build.

### 5c. The loudest one in a module this lane touched

It gets said plainly:

> **`survey_origin_budget.MAX_PAYLOAD_BYTES` (64 MiB) admits 87,838 households —
> 5.53% of source.** That is below one tenth, and below the 96,860-household
> preparation-receipt ceiling the transport lane lifted. A full-source payload is
> 1.13 GiB, 18.07× the cap.

Measured, not estimated: `experiments/native-row-ceilings/origin_budget_size.py`
builds one faithful origin record through the module's own `_reference` and
`_json` at full-source household-id widths and gets 764 bytes per group,
including the two `household_ids` and two `group_indices` entries the header
carries for each group's two clone roles.

**It also cannot be raised on its own, and that is a structural fact rather
than a preference.** `survey_origin_budget._json` is
`graph._bounded_json(value, MAX_PAYLOAD_BYTES)`, and `_bounded_json` opens with

```python
_require(type(limit) is int and 0 < limit <= 64 * 1024**2, "TRANSPORT_LIMIT")
```

so the shared encoder refuses any cap above 64 MiB before it encodes a byte. A
larger number in this module would not loosen the bound; it would refuse the
module. The 64 MiB is the encoder's, and moving it is the transport change.

The consequence for this lane is stated rather than glossed: **lifting
`MAX_GROUPS` is necessary and not sufficient.** `GROUP_COUNT_BOUND` is checked
in `_initial` before the payload is built, so at full source it fired first; with
it lifted, the refusal moves to `TRANSPORT_LIMIT` in the same module. That is the
correct outcome of one argument applied to one class of bound, and the remaining
ceiling is a real piece of work for whoever carries the segmented transport into
this module.

## 6. The census

`experiments/native-row-ceilings/census.json` is the census. Every `MAX_*` row,
household, person, group or byte bound in
`packages/microcosm-build/src/microcosm/build/us_runtime/` reachable from the
19-node financial graph and the 45-node pilot graph was read at this head — **146
bounds across five module families** — each with its enforcement site, its
refusal code and exception type, what it protects, the count it meets at 1/10
and at full source, and whether it binds. Every verdict that claimed a
full-source build meets the bound, or that the bound guards an encoding width or
an upstream file's real size, then went through an adversarial pass that read the
code again and tried to refute it — **41 verdicts**, and no headline verdict was
overturned.

The result: **14 bounds bind at full source, 6 of them at 1/10.** Seven of the
fourteen are the row counts §3 moved. The other seven are byte transports and
byte-derived row pre-checks; §5a and §5b say which and why.

## 7. Pins

These modules are hashed into stage implementation identities, so the pins were
re-derived through their generators rather than assumed.

| pin | generator | result |
|---|---|---|
| `graph_implementation_inventory.json` → `contracts` (all 124) | `graph_implementation._dependency_contract(payload, name, _covered_imports(name, inventory))` | unchanged by these edits — a constant's value is not an import, an unbound use or a resource access |
| `acs_native_coverage_binding._ACCEPTED["acs_pums.py"]` | `sha256(<module>.read_bytes())`, the same call the check makes | **moved**, `6ecf79f0…` → `e79a2a4e…` |
| the other three `_ACCEPTED` entries | same | unchanged; this branch does not edit those modules |
| all ten `implementation_manifest(stage)` | `graph_implementation.implementation_manifest` | built |

`experiments/native-row-ceilings/repin.py` reports all of this in one run and
`regenerate_accepted_pin.py` produced the moved value. Neither pin was
hand-edited, and both generators are idempotent.

What moves and is not a committed pin: each US stage's `implementation_hash` is
over its whole module roster, so editing any inventoried module moves it and
with it every node key and store address. #935 recorded that this is true of any
change to these files including a comment.
