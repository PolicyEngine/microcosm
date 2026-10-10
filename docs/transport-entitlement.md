# Entitlement declaration contract

This is an internal spec and hub hand-off contract. The executable example is
`test_support/microcosm_build/transport_entitlement.py`: its donor support,
facts, rates, references, pins and engine programs are invented test data.

The optional `entitlement_graph.json` resource has `schema_version: 1` and is
listed as a `legacy_json` row in the country package. Composition and the local
driver install it automatically. Its fields are:

| Field | Declaration |
| --- | --- |
| `nodes` | Ordered common bridge and validation node declarations, using the transport skeleton grammar. |
| `scenario_nodes`, `variant_nodes` | Node templates for the entitlement and calibration tiers of `scenarios.json`. A scenario row's `nodes` replaces its tier's template. Strings may contain `{scenario}` or `{variant}`. |
| `scenario_gap`, `variant_gap` | Gap producer id templates for the two tiers. |
| `bands` | `id`, `population`, `baseline_scenario`; all declared scenario gaps become typed inputs. |
| `package` | Final skeleton receipt `id`, bands `artifact_name`, optional additional `inputs` in the ordinary artifact-edge grammar. |
| `sources`, `checkpoints` | Additional sources with new names and ordered pre-export endpoints. The bands node is the default checkpoint. |

Activation checks resource selections in the common nodes and each scenario's
selected nodes, using the entitlement factory's row selection. A scenario's
explicit `nodes` replaces its tier template; an unused template contributes no
resource selections. Selected resources, paths and reference activation remain
preflight checks before registry preparation or graph-source reads.

Only explicit extension artifact edges may change a terminal skeleton package
receipt.
Every other skeleton predecessor set stays fixed. Scenarios branch below the
shared AS FILTER; calibration variants branch from the pre-calibration FILTER
and declare their own target/problem/calibration chain. AS inputs, AS engine
references and hold-out sources remain outside every calibration ancestor set.

Parameters accept the ordinary resource selectors and two additional selectors:

```json
{"binding": "rates_binding", "field": "engine_ref"}
{"binding": "rates_binding", "field": "period"}
{"scenario": "knobs", "path": [], "encoding": "json"}
{"scenario": "knobs", "path": [], "encoding": "sha256"}
```

Scenario selectors read the current row (`row` selects the entire row).
Their digest covers the selected subtree, so a sibling scenario edit or addition
does not move existing keys. Scenario overrides bind their selected knobs;
the keep-all scenario FILTER retains its key. Binding selectors inside selected
bridge JSON are resolved before canonical encoding. Structured literal records,
record lists and empty
lists are also canonical JSON; scalar lists remain graph tuples. Production
bridge declarations should select their coefficients and overrides from a
dedicated resource and bind its `resource_sha256`, rather than put policy data
in Python or bind the whole country fingerprint.

`simulate.counterfactual@1` uses a prepared engine-reference mapping. Primary
parameters are `engine_ref`, `period`, `variables`, `input_overrides` and
`output_coefficients`; `components` supplies independent additional engines.
Overrides name an entity/input column and a scalar value or source column.
Terms name a requested engine output, coefficient and target column. A
person-to-family term declares `sum`, `max` or `first` aggregation; masks select
receipt branches. Components evaluate on independent copies of the input frame.
All bridge outputs are produced float64 columns. Each family engine node and
keep-all FILTER declares a person data-column slice.

`simulate.solve_zero@1` adds `solve_input`, `bracket`, `tolerance` and
`iterations`. Each output row's residual must be independent of other rows'
trial inputs and non-increasing in its own trial input over the bracket. The
kernel does not support coupled systems. Fixed-count vector bisection locates
the boundary of positive residuals, including a benefit's zero plateau;
an already nonpositive lower endpoint returns that endpoint. A residual of
exactly zero at a positive trial input retains the nonzero boundary answer.
The iteration count must achieve the declared input-interval tolerance in
float64. After the fixed iterations the kernel evaluates the returned answers
and refuses if any row's positive residual exceeds the same declared tolerance.
A successful run reports `iterations + 3` residual evaluations: two bracket
endpoints, the fixed iterations and final verification. Each component engine
runs once per residual evaluation. The caller supplies the reference-case
overrides.

`takeup.gap@1` parameters are `country`, `references`, `references_sha256`,
`entities`, `weight_entity`, `annualisation_factors` and `scenario` (optional
`facts_sha256` and `annualisation_sha256`). References select measured columns,
filters, comparator facts and table-cell metadata. Metadata labels are flat
string-valued fields, following the target compiler contract. Annualisation
factors map each reference name to its positive factor; count measures can
declare a factor of one. The artifact
records weighted entitlement `E`, published actual `A`, exactly `E - A`, the
scenario and reference/fact trace. Groupings, including decile by family type
and receipt strata, are reference data. `takeup.bands@1` requires identical
table definitions across scenarios, records low/baseline/high for E, A and gap,
and names each bound's scenario; ties use the first label in lexical order.

The NZ package remains inactive. Its activation owner must supply the graph
declaration, prepared engine and RuleSpec pins, executable bridge overrides,
non-null knobs, active comparison references and annualisation/grouping data.
V1/V2 need explicit aged target-reference resources: the current target compiler
does not accept an `aging_index` parameter. If pre-calibration quantile maps are
aged too, those variants must branch from CREATE. V3 declares its receipt
override before its independent problem and calibration.

Engine components and masks can declare an NZS/VP alternative; an explicit
exclusion input and counted excluded-unit reference can declare the fallback.
The toy tests exercise these declaration and grouping shapes with invented
rules. Actual reg 17 treatment requires approved bridge data, legal-source
review and real-engine differential evidence.

Real WFF remains deferred pending approved input encoding, RuleSpec bindings
and Child-rule treatment. The supplied graph plan describes three chained
modules; this checkout does not certify their law semantics or external
approval. A tripwire declaration must calculate the WFF measure before
comparing it with the hold-out reference. The toy graph
does this with its invented family-credit engine. It supplies no real WFF
inputs or statutory claim.

The committed `build/nz/as_rate_bridge.json` sole-parent declaration rules out
a `solve_zero` recipe for golden-08: no engine input combination reproduces
the harness's single-with-children gross rate paired with the Income Test 3
slope. It instead declares three evaluations, `gross_rate`,
`reduction_at_anchor` and `reduction_after_step`, whose outputs recover the
slope and lower threshold before assembling the cutout. The resource declares
the overrides, anchor, step, composition and refusal conditions. The driver
executes declared kernel calls but does not assemble this nonlinear
composition. The approved executable golden-08 case and evaluation assembly
therefore remain unresolved; no golden-08 result is claimed.

The hub can compare approved small cases and sampled-unit outputs without donor
H5 I/O using the declared case format in
`tools/transport_bridge_differential.py`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$(ls -d $PWD/packages/*/src | tr '\n' ':')$PWD:$PWD/test_support" \
UV_CACHE_DIR=$PWD/.hub-scratch/uv-cache \
uv run --no-project .venv/bin/python tools/transport_bridge_differential.py \
  --case <approved-small-case.json> \
  --oracle <axiom-oracles/conformance/executable/nz-treasury-incomeexplorer/requests.json> \
  --rulespec-root <exported-pinned-rulespec-tree> \
  --out .hub-scratch/bridge-receipt.json
```

The case supplies unit-rule data, bindings, small entity tables and weights,
ordered kernel calls, and oracle JSON pointers with explicit absolute
tolerances. Call IDs must be distinct and nonempty, each output entity/column
coordinate must have one producer, and comparisons must name a declared call
output. These checks precede execution. Oracle pointers use RFC 6901 escapes;
array indices must be canonical nonnegative integers within bounds. A case can
include sampled units and matching oracle arrays. With the engine available,
the receipt records each comparison and the case/oracle hashes. If the engine
is absent the command writes a skipped receipt without reading case data;
no real-engine result is claimed.
