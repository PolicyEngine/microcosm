"""The build's constants name what the UK engine-population declaration does.

microcosm-frame cannot import microcosm-build, so the two things both name
are held together here (microcosm#931): the household column spine engine
calls batch by, which must be the ladder frame's clone index, and the frame
columns each share formula's allocation key adds, which the measure step
sums for the pool-share correction.
"""

from __future__ import annotations

from microcosm.build.uk_runtime.full_measure import UK_WEIGHT_SHARE_FORMULA_INPUTS
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column
from microcosm.frame.adapters.policyengine_uk_population import (
    UK_ENGINE_POPULATION,
    UK_SPINE_ENGINE_BATCH_COLUMN,
)


def test_spine_engine_calls_batch_by_the_ladder_households_clone_index():
    assert UK_SPINE_ENGINE_BATCH_COLUMN == ladder_clone_index_column("household")


def test_the_share_inputs_the_build_sums_are_the_declared_key_adds():
    declared = {
        formula: UK_ENGINE_POPULATION.key_adds[key]
        for formula, key in UK_ENGINE_POPULATION.corrected.items()
    }
    # Tuples: the same columns in the same order.
    assert dict(UK_WEIGHT_SHARE_FORMULA_INPUTS) == declared
