# US raw-PUF growth: first slice

The first bounded increment of [microcosm#530](https://github.com/PolicyEngine/microcosm/issues/530)
item 2 — *pin the raw PUF asset and implement PUF aging as a Microcosm stage,
under the same uprating discipline as the other sources*. It adds a pure
per-field growth contract, the transform that applies it, and three graph nodes
that run it. It adds no source pin, no artifact, no factor value, and no build
wiring, because none of those exists yet in an authenticated form.

Read this together with the two evidence reports it is built on: the PUF
producer-provenance audit and the non-ASEC donor monetary-basis audit
(`/Users/maxghenis/architecture-reviews/microcosm-20260904/`). Their conclusion
governs here: **the price basis and production chain of the pinned processed
`puf_2024.h5` remain unverified**, so that artifact stays an opaque legacy
comparison input and is not the root of anything below.

## What this slice contains

| File | What it is |
| --- | --- |
| `packages/microcosm-build/src/microcosm/build/us_runtime/puf_growth.py` | The contract types, the closed-coverage recipe, the factor table, and the pure transform. Reads no file. |
| `packages/microcosm-build/src/microcosm/build/us_runtime/puf_growth_graph.py` | Three nodes and three kernels that run the transform inside a real population. |
| `packages/microcosm-build/tests/test_us_puf_growth.py` | 48 contract tests on invented rows and invented factor levels. |
| `packages/microcosm-build/tests/test_us_puf_growth_graph.py` | 8 tests: a real compiled graph, cold / warm / `resume="require"` / fresh-store parity against a direct-call oracle. |

Nothing was added to `us_runtime/__init__.py`; both modules are imported by
path, as `asec_current_money` and its siblings are. Importing
`microcosm.build.us_runtime` was measured in this environment: it succeeds and
loads **no** `policyengine*` module, because every PolicyEngine import in that
package — including `microcosm.frame.adapters.policyengine_us` — is lazy. So
placing these files in `us_runtime` does not drag in a country engine.

## The contract

A **rule** covers one field, or one sign branch of one field. It declares:

| Declaration | Meaning |
| --- | --- |
| `field` | The raw source column, e.g. `E00900`. |
| `role` | `money`, `code`, `identifier`, `count`, or `design_weight`. |
| `source_reference_year` | The year the observed amount belongs to. |
| `target_year` | The year the output amount is stated in. |
| `growth_kind` | `nominal_price_restatement`, `real_per_return_aging`, `design_weight_growth`, or `none`. |
| `sign_branch` | `any`, `positive`, or `negative`. |
| `factor_series` | The exact variable-row name in the factor table. |

`nominal_price_restatement` and `real_per_return_aging` are kept apart on
purpose. The first restates the same nominal amount into another year's
purchasing power. The second is the SOI-style operation the archived producer
used — an aggregate-total ratio divided by a return-count ratio — which carries
real change as well as price change. They are not interchangeable, and nothing
recovers which one a bare number was.

A **recipe** is a closed set of rules, validated before any table is seen:

- no rule may take a provenance column's name, so the provenance write cannot
  silently overwrite a declared field;
- exactly one identifier rule, and a money rule for the source-year AGI, because
  both provenance columns are mandatory;
- no duplicate `(field, sign_branch)`;
- a field is covered either by one `any` rule, or by exactly one `positive` and
  one `negative` rule — never both shapes, never half a sign cover;
- every rule for a field agrees about that field's role, years and growth kind;
- all money fields share one target year and one source reference year;
- at least one money field, and at most one design weight, whose target year
  matches.

`require_transformable` re-checks those preconditions without a table, so a node
factory refuses a contract that could never run at declaration time rather than
after a graph run has already loaded a source.

A **factor table** is a document of *index levels*: variable rows by year
columns, plus an authority and a provenance block. A factor is always
`level(row, target_year) / level(row, source_year)` — two explicitly named year
columns of one explicitly named variable row. There is no default of 1.0 and no
substring dispatch: an unknown row refuses with `UNKNOWN_FACTOR_SERIES`, a
missing year column with `MISSING_FACTOR_YEAR`, and a row whose declared kind
disagrees with the rule's with `FACTOR_KIND_MISMATCH`.

Compiling a recipe against a table resolves each factor exactly once and freezes
it, including the exact float64 bit pattern, into the contract's identity bytes.

## The two archived defects, and why they cannot recur here

Both were established by reading the retired US data package at commit
`42ed5d45c56df80d754fbe24cce21cfeb8d05cbe` from its local git object database.
This repository's launch contract forbids naming that package in the live tree
(`test_no_incumbent_data_package_references_in_live_tree` in
`packages/microcosm-build/tests/test_us_plan.py`), so it is identified here and
in `puf_growth.ARCHIVED_PRODUCER_COMMIT` by that immutable commit alone; the
paths quoted below are relative to its Python package directory.

**Year columns iterated as if they were variable names.**
`utils/uprating.py` pivots to `index="Variable", columns="Year"`, so the frame's
column labels are integer years. `datasets/puf/puf.py` then does
`for variable in uprating:` — `DataFrame.__iter__`, which yields column labels.
`variable` is therefore a year, `if variable in arrays` is false on every
iteration, and no growth is applied at all. (The loop body is independently
broken too: `uprating.Variable` raises `AttributeError`, because `Variable` is
the index name, not a column.)

Here the two axes are different types with different lookups: a variable row is
found by exact name in a name-keyed mapping, a year is found by exact value in a
year-keyed mapping, and neither can stand in for the other. The test
`test_year_columns_are_not_variable_rows` asserts the row names are disjoint
from the year labels, that asking for `"2024"` as a series refuses, and that a
recipe naming a year as its `factor_series` refuses.

**One field multiplied twice.**
`datasets/puf/uprate_puf.py` lists `E00900` and `E26270` in a sign-specific
rename dict *and* in the AGI-growth fallback list, so a row of either field
received a sign-specific factor and then a second AGI factor. (`E01000` is in
the sign dicts but not the fallback list, so it was correctly single-treated.)

Here that recipe cannot be constructed: a field with a `positive`/`negative`
pair plus an `any` rule refuses with `OVERLAPPING_SIGN_RULES` before anything is
computed. On top of the structural refusal, the transform counts applications
per row and refuses `FACTOR_APPLIED_TWICE` if any row were ever selected twice.

## Preservation

- Codes, identifiers and counts are **declared** and never written. Two executor
  mechanisms keep them byte-identical, and it is worth naming them exactly: a
  kernel's returned coordinates must equal the node's declared outputs, so it
  cannot write those columns at all; and the projected input view is frozen and
  its digest compared before and after the kernel runs, so it cannot mutate them
  in place either. (There is no non-owned *storage-equality* check on an
  ordinary node — that path exists only for `EXPAND`.)
- Money growth and design-weight growth are separate entry points, separate
  graph nodes, and **separate column sets**. The design weight is not in the
  money view at all: `apply_puf_growth` refuses a table that contains it
  (`DESIGN_WEIGHT_NOT_IN_MONEY_VIEW`), and the money node does not declare it as
  an input. That matters in the graph — if the money node read the weight
  column, the executor would make the weight node its predecessor and a
  weight-only factor change would recompute every money column.
- Zeros are never selected, so a zero never enters arithmetic. A source negative
  zero is refused outright (`NEGATIVE_ZERO`) rather than carried through, so the
  grown column cannot hold a bit pattern that compares equal to zero while
  hashing differently.
- A money column must be an integer or float kind — complex is refused rather
  than silently losing its imaginary part — and an integer amount larger than
  `2**53` is refused rather than rounded by the float64 working conversion.
- Every product is checked: `GROWN_NONFINITE` and `SIGN_CHANGED` on money,
  `GROWN_WEIGHT_NONFINITE` and `GROWN_WEIGHT_NONPOSITIVE` on weights.
- The input frame is never mutated; every refusal happens before any array is
  written.
- Raw `RECID` and the source-year AGI are copied into `puf_source_recid` and
  `puf_source_year_agi` **before** any growth, so a later change to `E00100`'s
  basis cannot destroy the input to the TY2015 SOI band assignment that
  `puf_source_agi.py` and `puf_interest_components.py` depend on. "Immutable"
  here means contract-enforced: the transform writes them once from pre-growth
  values and no rule may target them.

## Monetary metadata

`puf_growth_monetary_metadata` returns, per money field, the existing
`microcosm.build.monetary_targets.MonetaryBasis` — currency, target period,
`annual_flow`, and a valuation string that names the growth kind and the source
year (`real_per_return_aged_from_2015`, `nominal_price_restated_from_2015`).
That is the vocabulary a future donor/recipient fit-boundary check needs. **The
check itself is not implemented here**, and no existing consumer path was
changed to consume this.

## Authority

A factor table declares its own authority. `invented_fixture_nonauthority` is
the only value that compiles today, and a contract built from one reports
`release_eligible: False` in every receipt. `reviewed_resource` refuses with
`UNREVIEWED_FACTOR_TABLE` — on the compiled type itself, not only in the compile
function, so no construction path yields a release-eligible contract — because
`REVIEWED_FACTOR_TABLE_DIGESTS` is empty:
no factor table for this lineage has been generated or reviewed. Adding a digest
there is the reviewed act; there is no flag a caller can pass instead. A
reviewed table would additionally have to state its generation receipt
(`REVIEWED_PROVENANCE_KEYS`: generator code digest, dependency versions,
parameter paths, index-or-total, population divisor, rounding, source URL).

**Every factor number in this repository's PUF growth fixtures is invented.**
They exist to prove the contract and the transform. They are not IRS values, not
SOI values, not CPI values, and not a release recipe.

## The field roster

`PROPOSED_TY2015_FIELD_ROSTER` names real raw PUF field codes with a role and a
documented meaning, and **declares no factor for any field**. The meanings come
from the archived `datasets/puf/aggregate_record_totals.yaml`, whose own header
cites the *2014 PUF General Description Booklet*, pages 33–36 (June 2020). That
booklet was not read here, so a roster meaning is "the label the archived
producer attached to this code", not an independently authenticated IRS
definition. `RECID`, `S006`, `MARS` and `XTOT` are cited to `preprocess_puf` at
the same commit; `P22250` to `aggregate_record_utils.SCREENED_FIELDS`.

The roster is a **lower bound**: every money code the archived codebook
describes, plus the four structural codes and two codes cited separately
(`P22250` from the screened-field list, `E02500` from the SOI rename table). It
is not every code this repository reads and not every code the delivered file
contains. `UNDOCUMENTED_SOURCE_FIELDS` records three codes the archived commits
name in executable code without ever describing — `DSI`, `EIC` and `E25920` —
and they are deliberately absent from the roster. That list is what this slice
tripped over, not a complete inventory of undocumented codes.

## The graph nodes

```
<stage>.boundary        FILTER   compiles the contract, keeps every person,
                                 opens a version so declared money columns
                                 can be rewritten
<stage>.money           COMPUTE  reads every declared field except the design
                                 weight; owns each money column (rewrite) plus
                                 the two provenance columns
<stage>.design_weight   COMPUTE  reads and owns only the raw design-weight
                                 column
```

The contract travels as node parameters — the recipe identity document and the
factor document, verbatim — so a changed level moves the node key rather than
being served from the store. `test_a_changed_factor_level_moves_the_node_key_and_is_not_reused`
asserts exactly that.

The design weight this stage grows is the **raw source column** (`S006`). The
frame's typed design weights are left exactly as the source loaded them;
promoting the grown column into typed weights is a separate obligation with its
own mass policy (below).

## Running it

```bash
uv sync --all-packages --locked          # no us/uk extras are needed
uv run pytest packages/microcosm-build/tests/test_us_puf_growth.py \
              packages/microcosm-build/tests/test_us_puf_growth_graph.py   # 56 tests
uv run ruff check .
uv run python tools/ci_test_groups.py --verify
```

Both test files sit flat in `packages/microcosm-build/tests/` and are named
`test_us_p…`, so they land in the `us-p` engine group and the `rest` fast group,
and never under `[defaulted]`. Neither needs a country engine.

## Remaining obligations

Nothing below is done, and none of it is implied by anything above.

1. **Authenticate a raw source.** The licensed raw TY2015 PUF and the
   demographic inputs must be authenticated by metadata/receipt and pinned
   before any real build. This slice adds no `source_stages.json` entry and no
   `us/spec/sources.yaml` pin, deliberately.
2. **Generate a reviewed factor table.** Its receipt must record the exact
   generation code, the PolicyEngine-US/Core pins, the parameter paths, the
   index-versus-total classification, the population divisor, the rounding
   precision, the source and target years, and the resulting hash — which is
   then added to `REVIEWED_FACTOR_TABLE_DIGESTS`. Neither historical CSV
   (`uprating_factors.csv`, the local growth CSV) may be adopted as that table
   without that review.
3. **Decide the per-field growth policy.** Which fields are price restatement
   and which are real per-return aging is a producer decision, not an
   inheritance. In particular: whether the archived 2%/yr itemized-deduction
   assumption for `E17500`, `E18400`, `E18500`, `E19200`, `E19800` is retained,
   replaced or dropped; and what happens to the fields the archived producer
   consumed but never uprated at all — `E00100`, `E02100`, `E25920`, `E25940`,
   `E25960`, `E25980`, `E26180`, `E26190`, `E30400`, `E30500`.
4. **Resolve the raw-source field manifest.** Close the roster against real
   source documentation: whether `E22250` and `P22250` both exist in the
   delivered file, what `DSI`, `EIC` and `E25920` are, and — the larger gap —
   which codes the delivered file contains that no archived code path touched,
   since nothing read here can see those.
   Relatedly: the graph stage's closed read set is the *transform's*, not the
   frame's. A `CREATE` node may load raw money columns this contract never
   declares, and they pass through ungrown. Closing that needs a source-stage
   contract that declares the full raw column inventory.
5. **Decide one-step versus two-step aging.** The archived chain was
   TY2015→2021 from SOI aggregates, then 2021→target from model uprating
   factors. A rule here declares one `(source, target)` pair, so composing two
   steps means either a composed level series or two declared stages with an
   intermediate artifact. Either is a reviewed choice; neither is implemented.
6. **Promote the grown design weight into typed frame weights**, with an
   explicit `WeightTransition` and mass policy. Only the raw column grows today.
7. **Person expansion.** The demonstration frame carries one filer person per
   return, which is a fixture placeholder, not a modelling claim. A real
   successor must decide how `MARS`, `XTOT`, `AGERANGE`, `AGEDP1/2/3`, `GENDER`
   and `EARNSPLIT` become persons.
8. **Excluded from this slice by instruction, and still owed**: the Section 199A
   / QBI simulation port (#530 item 3), the carried-leaf port (#530 item 4), the
   Forbes-tail policy, the IRS disclosure aggregate-record disaggregation, raw
   artifact authentication, and genuine factor generation.
9. **The fit-boundary compatibility check itself.** The metadata exists; the
   comparison between a donor field's basis and a recipient predictor's basis
   does not.
10. **Comparison with the pinned processed artifact stays diagnostic.** Exact
    parity with a possibly defective price basis is evidence, not an acceptance
    rule, per the provenance audit.

## What this slice did not verify

No PUF was read. No `.h5`, `.dta` or survey CSV was opened. No build, fit, model
calculation, resource generation, or network asset download was run. Nothing
here establishes what the pinned `puf_2024.h5` contains, what price basis any of
its fields carry, or which code produced it. The archived-producer facts above
were read from the retired US data package's local git object database at the
commit above; the IRS booklet those archived comments cite was not read.
