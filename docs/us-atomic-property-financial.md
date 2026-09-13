# Current survey property-income host

`run_atomic_survey_financial` can append the current property-income graph to
the existing financial composition. Pass an explicit `PropertyIncomeOptions`
through `property_income`; omitting it retains the nineteen-node path.

```python
from microcosm.build.us_runtime.graph_atomic_survey_financial import (
    run_atomic_survey_financial,
)
from microcosm.build.us_runtime.graph_current_survey_property import (
    PropertyIncomeOptions,
)

options = PropertyIncomeOptions(
    scales=(1.0, 1.0, 1.0, 1.0),
    atol=1e-10,
    rtol=1e-12,
    n_estimators=2,
)
run = run_atomic_survey_financial(
    **source_arguments,
    geography_config=atomic_geography,
    property_income=options,
    n_estimators=2,
    return_values=True,
)
checked = run.checked_view()
```

The small tree counts above are development settings. The caller supplies the
existing authenticated source arguments and atomic geography configuration.
They are not replaced by a downloaded graph, a detached Frame or a receipt.

The thirty-five-node composition consists of the nine-node survey, clone and
atomic-geography prefix; ten existing financial nodes; and sixteen property
nodes. The new branches select eligible original ASEC donors and original ACS
recipients before allocation. Four ordered, DESIGN-weighted conditional models
produce raw component draws, followed by reconciliation to a known adult ACS
property-income anchor. An explicit attachment carries each original result to
both initial clones. Original ASEC components remain separately qualified even
when that person is excluded from fitting. See the
[fragment contract](us-current-survey-property-graph.md).

The output adds 24 component, raw-draw, diagnostic and knownness columns. It
preserves the complete preceding financial population: other columns, entity
membership, geography, schema, metadata, weights, strata, ownership and mass
history. Missing source anchors and unresolved components retain missing
values and explicit knownness. This mode does not infer property-income zeros
for children outside the survey question's universe.

The runner retains both the preceding and extended populations. Before issuance,
execution and required cache replay verify property model training against the
original donor and replay four applications against actual recipient features.
Later issued-run checks requalify the existing source owners, validate the legacy
financial output, reconstruct the complete extended population and check the
verified model artifacts' byte identities. They do not refit or reapply the
models. Copied public result dataclasses have no run authority. See the
[model receipt contract](us-property-model-receipts.md).

The thirty-five-node option leaves the eight existing financial leaves unchanged.
Its run document therefore records `tax_split_rebased: false` and the legacy
capital-gain model's conditioning on its earlier interest and dividend draws.
It does not itself replace tax inputs. Optional tax rebasing is a separate
three-node extension, described below.

The PUF host derives its upstream node count and checks the actual final
financial writer. Its recipient nodes read all columns of that population,
including the new property fields, so the dependency is present in the graph.
This interface does not establish that a native PUF run, tax rebase, calibration
or release has passed. Candidate execution evidence belongs in `experiments/`.

## Optional tax rebase

Pass `rebase_property_taxes=True` together with `property_income=options` to
append the deterministic receiving version, tax split and numerical gate. The
default is `False`; requesting a rebase without property options refuses before
source I/O. This path has 38 nodes. It retains the complete legacy financial
population and the complete 35-node property population separately from the
final `survey_property.tax_receiving` version.

The rebase uses ordinary interest and dividends with the maintained fractions
and subtraction complements. Separate missing O/D inputs produce missing tax
leaves, replacing earlier predictions; retirement-account interest remains
auxiliary. A new FILTER conservation record is intentional, while existing
Frame mass history and all non-rebased cells remain unchanged. See the
[numerical fragment contract](us-property-tax-leaves.md).

The host constructs the declarations from the checked clone Frame and the
existing financial/property output descriptors. It never supplies fake values
for future columns. After the real 35-node output is available, it independently
re-declares and executes the three deterministic operations, checks stored
artifact bytes and complete node receipts, and compares every expected full
population with actual cold or cached observations. The issued run retains
the source, declaration, model, artifact, intermediate population and final
population seals. Later checks reconstruct the same tax operations before the
final source requalification and pure lifetime fence.

An incomplete numerical gate is valid development evidence. The run document
reports `tax_split_rebased: true`, three extra nodes and the actual
`tax_leaf_complete` value; it remains `release_eligible: false`. PUF recipient
qualification explicitly requires a complete independently checked numerical
gate before it reads its additional source inputs. Both PUF recipient graph
nodes carry a typed edge to that gate on the actual receiving version. The
19- and 35-node paths do not add that edge or claim a tax rebase.

No child zero completion or missing-adult model is introduced. Consequently the
invented source fixture with under-15 and missing-anchor cases must remain a
development run and refuse full PUF qualification. Capital gains retain their
earlier conditioning; rebasing interest and dividends does not update that model.

The bounded invented-source acceptance covers 11 host controls, including actual
cold and required-cache 38-node runs, issued-view parity, intermediate/final
population preservation, mutation refusals and incomplete PUF qualification.
The numerical fragment has a separate 40-control acceptance, including a
complete-input positive gate and descriptor-only declaration checks. These
checks do not use native survey data, run a country engine or execute full PUF.
