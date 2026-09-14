"""Historical budget pair consumers over actual, invented survey sources."""

import hashlib
import json

from test_us_graph_survey_population import authenticated_arguments

from microcosm.build.us_runtime import current_survey_predictors as predictors
from microcosm.build.us_runtime import graph_puf_diagnostic_consumer as diagnostic
from microcosm.build.us_runtime import graph_survey_population as graph
from microcosm.build.us_runtime import survey_origin_budget as budget


def test_original_budget_pair_keeps_current_survey_qualifiers_compatible(
    tmp_path, monkeypatch
):
    run = graph.run_authenticated_survey_population(
        **authenticated_arguments(tmp_path, monkeypatch),
        clones=True,
        return_values=True,
    )
    raw, clone = run.allocated_population, run.clone_population
    identities = tuple(budget._population_identity(p) for p in (raw, clone))
    view = run.preparation.checked_view()
    historical = budget._initial(view, raw, clone)
    assert type(historical) is tuple and len(historical) == 2
    instructions, allocation = historical
    assert len(instructions) == raw.frame.n("household")
    assert budget._initial(view, raw, clone, _with_geography_binding=True) == (
        *historical,
        None,
    )
    allocation_sha256 = hashlib.sha256(allocation).hexdigest()
    # Execute both existing pair consumers, without fitting a model or assigning
    # a PUF outcome, Social Security beneficiary, or conditioning total.
    qualified = predictors.qualify_current_survey_predictors(
        run.preparation, raw, clone
    )
    assert qualified.evidence["allocation_sha256"] == allocation_sha256
    projection, _matrix, _mask, _evidence = diagnostic.qualify_current_survey_host(
        run.preparation, raw, clone
    )
    assert json.loads(projection)["allocation_sha256"] == allocation_sha256
    assert tuple(budget._population_identity(p) for p in (raw, clone)) == identities
