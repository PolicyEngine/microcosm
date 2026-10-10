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
`iterations`. It uses fixed-count vector bisection of a declared non-increasing
residual and returns the first nonpositive upper boundary, including a floored
benefit's zero plateau. The iteration count must achieve the declared bracket
tolerance in float64. The caller supplies the reference-case overrides; for
golden-08 the cutout uses JSS single-with-children inputs, independently of the
unit's actual benefit.

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

The NZS/VP reg 17 treatment is declared through engine components and masks;
the implementation contains no statutory branch. The alternative is an
explicit exclusion input and a counted excluded-unit reference. The toy tests
exercise an engine-derived alternative and exclusion/grouping shape. Actual
NZS/VP correctness requires the hub's approved bridge data.

Real WFF stays parked until its input encoding, three chained module bindings
and Child-rule treatment are approved. A tripwire declaration must calculate
the WFF measure before comparing it with the hold-out reference. The toy graph
does this with its invented family-credit engine. It supplies no real WFF
inputs or statutory claim.

The hub can run golden-08 and small sampled-unit differentials without donor H5
I/O using the declared case format in `tools/transport_bridge_differential.py`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$(ls -d $PWD/packages/*/src | tr '\n' ':')$PWD:$PWD/test_support" \
UV_CACHE_DIR=$PWD/.hub-scratch/uv-cache \
uv run --no-project .venv/bin/python tools/transport_bridge_differential.py \
  --case <approved-small-case.json> \
  --oracle <axiom-oracles/conformance/executable/nz-treasury-incomeexplorer/requests.json> \
  --rulespec-root <exported-pinned-rulespec-tree> \
  --out .hub-scratch/golden08-receipt.json
```

The case supplies unit-rule data, bindings, small entity tables and weights,
ordered base-rate/cutout/AS calls, and oracle JSON pointers with explicit
absolute tolerances. It can include sampled units and matching oracle arrays.
The receipt records each comparison and the input hashes. If the engine is
absent the command writes a skipped receipt; no real-engine result is claimed.
