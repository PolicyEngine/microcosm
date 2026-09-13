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

This first extension leaves the eight existing financial leaves unchanged.
Its run document therefore records `tax_split_rebased: false` and the legacy
capital-gain model's conditioning on its earlier interest and dividend draws.
The new property components do not yet establish coherent replacement tax
inputs. A later rebase must use ordinary interest rather than account earnings,
preserve separately known ASEC observations, and explicitly complete any
required unknown inputs before full PUF use.

The PUF host derives its upstream node count and checks the actual final
financial writer. Its recipient nodes read all columns of that population,
including the new property fields, so the dependency is present in the graph.
This interface does not establish that a native PUF run, tax rebase, calibration
or release has passed. Candidate execution evidence belongs in `experiments/`.
