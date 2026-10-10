"""A per-clone engine block scaled to the pool measures weight-share formulas exactly.

policyengine-uk allocates the ONS aggregate corporate land value across households by
``corporate_sector_wealth * weight / sum(corporate_sector_wealth * weight)``. One engine over
a block that carries a K-th of the pool's mass hands that block the whole national total (the
K-times artefact behind the #736 erratum); the same block with its engine weights scaled by K
reproduces the pool's values. A formula without a weighted denominator is untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind


def _frame(weights):
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 10, 20, 20],
            "person_household_id": [100, 100, 200, 200],
            "age": [35, 8, 40, 38],
            "is_benunit_head": [True, False, True, False],
            "is_parent": [True, False, False, False],
            "is_uc_claimant": [False, False, False, False],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [10, 20],
            "benunit_source_id": [10, 20],
            "dependent_children": [1, 0],
            "is_married": [False, False],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_source_id": [100, 200],
            "region": ["LONDON", "WALES"],
            "council_tax": [1800.0, 900.0],
            "rent": [0.0, 6000.0],
            "tenure_type": ["OWNED_OUTRIGHT", "RENT_PRIVATELY"],
            "corporate_wealth": [120_000.0, 15_000.0],
            "private_pension_wealth": [80_000.0, 0.0],
            "property_wealth": [650_000.0, 0.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=2024,
        household_weights=np.asarray(weights, dtype=float),
        weight_kind=WeightKind.IMPORTANCE,
    )


def _values(resolver, variable):
    return np.asarray(resolver.simulation.calculate(variable, 2025), dtype=np.float64)


def test_block_engine_scaled_to_the_pool_reproduces_the_pool_land_value(tmp_path):
    pytest.importorskip("tables")
    pool = _frame([1000.0, 500.0])
    # one of four identical clones: the same households at a quarter of the mass
    block = _frame([250.0, 125.0])

    pool_resolver = UKMeasureResolver(
        simulation_source=None, frame=pool, year=2025, scratch_dir=tmp_path / "pool"
    )
    naive_resolver = UKMeasureResolver(
        simulation_source=None, frame=block, year=2025, scratch_dir=tmp_path / "naive"
    )
    scaled_resolver = UKMeasureResolver(
        simulation_source=None,
        frame=block,
        year=2025,
        scratch_dir=tmp_path / "scaled",
        engine_weight_scale=4.0,
    )

    pool_value = _values(pool_resolver, "corporate_land_value")
    assert pool_value.sum() > 0.0
    # the artefact: the block alone reproduces the whole national aggregate
    np.testing.assert_allclose(
        _values(naive_resolver, "corporate_land_value"), 4.0 * pool_value, rtol=1e-6
    )
    # the representation: scaled to the pool, the block measures the pool's values
    np.testing.assert_allclose(
        _values(scaled_resolver, "corporate_land_value"), pool_value, rtol=1e-6
    )
    np.testing.assert_allclose(
        _values(scaled_resolver, "land_value"),
        _values(pool_resolver, "land_value"),
        rtol=1e-6,
    )
    # a formula without a weighted denominator never depended on the block's mass
    np.testing.assert_allclose(
        _values(naive_resolver, "household_land_value"),
        _values(pool_resolver, "household_land_value"),
        rtol=1e-9,
    )
    assert scaled_resolver.receipt()["engine_weight_scale"] == 4.0
    assert "engine_weight_scale" not in naive_resolver.receipt()
