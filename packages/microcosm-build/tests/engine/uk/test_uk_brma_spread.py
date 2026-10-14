"""Small UK-engine contract for BRMA-spread mixtures and rate ordering."""

import numpy as np
import pandas as pd

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.brma_spread import lha_rate_keys, prepare_brma_spread
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine


def test_engine_materializes_household_mean_lha_rates_for_each_regional_brma():
    # Two differently sized units in the London household make aggregation
    # across benefit units observable; Scotland has only one BRMA slot.
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 20, 20, 30],
            "person_household_id": [1, 1, 1, 2],
            "age": [40, 40, 39, 40],
            "is_benunit_head": [True, True, False, True],
            "is_parent": True,
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": [10, 20, 30], "LHA_category": ["B", "C", "B"]}
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["LONDON", "SCOTLAND"],
            "brma": ["CENTRAL_LONDON", "LOTHIAN"],
            "tenure_type": "RENT_PRIVATELY",
            "rent": 12_000.0,
            # The single-year loader extends and uprates the dataset even
            # for a same-year LHA calculation, requiring council tax input.
            "council_tax": 0.0,
        }
    )

    def frame_for(table):
        return uk_national_frame(
            person=person,
            benunit=benunit,
            household=table,
            household_weights=[2.0, 3.0],
            time_period=2025,
        )

    engine = PolicyEngineUKEngine()
    frame = frame_for(household)
    original = {entity: frame.table(entity).copy() for entity in frame.entities}
    original_weights = frame.weights_for("household").values.copy()
    brmas = {
        "LONDON": ["CENTRAL_LONDON", "INNER_EAST_LONDON"],
        "SCOTLAND": ["LOTHIAN"],
    }
    keys = lha_rate_keys(frame, engine=engine, brmas=brmas)
    variable = "uncapped_BRMA_LHA_rate"
    try:
        engine.variable_metadata(variable)
    except ValueError:
        variable = "BRMA_LHA_rate"
    assert variable not in engine.variables()
    for slot, london_brma in enumerate(brmas["LONDON"]):
        moved = household.assign(brma=[london_brma, "LOTHIAN"])
        rate = engine.materialize(frame_for(moved), [variable], 2025)[variable]
        assert np.isfinite(rate).all()
        assert rate[0] != rate[1]
        np.testing.assert_allclose(keys[0, slot], rate[:2].mean())
        if slot == 0:
            np.testing.assert_allclose(keys[1, slot], rate[2])
    assert keys[1, 1] == np.inf

    # The engine's actual category materialization must select B/C/B census
    # cells. Different cell totals distinguish averaging normalized unit
    # distributions from pooling raw counts or averaging across people.
    resource = {
        "cells": {
            "LONDON": {
                "B": {"CENTRAL_LONDON": 1, "INNER_EAST_LONDON": 3},
                "C": {"CENTRAL_LONDON": 9, "INNER_EAST_LONDON": 1},
            },
            "SCOTLAND": {"B": {"LOTHIAN": 7}},
        }
    }
    lineage_keys = [
        household_draw_key(
            source="frs",
            source_vintage="2023_24",
            source_household_id=household_id,
            clone_path=(),
        )
        for household_id in (1, 2)
    ]
    payload = prepare_brma_spread(
        frame, engine=engine, lineage_keys=lineage_keys, count_resource=resource
    )
    assert payload["household_ids"] == [1, 2]
    assert payload["ordered_brmas"][0] == [
        brmas["LONDON"][index] for index in np.argsort(keys[0], kind="stable")
    ]
    known_probabilities = {
        "CENTRAL_LONDON": (1 / 4 + 9 / 10) / 2,
        "INNER_EAST_LONDON": (3 / 4 + 1 / 10) / 2,
    }
    probabilities = np.diff(
        np.asarray(payload["cumulative_probabilities"]), axis=1, prepend=0.0
    )
    np.testing.assert_allclose(
        probabilities[0],
        [known_probabilities[name] for name in payload["ordered_brmas"][0]],
        rtol=0,
        atol=1e-12,
    )
    # Scotland's sole supported BRMA precedes a safely padded zero column.
    assert payload["ordered_brmas"][1] == ["LOTHIAN", "LOTHIAN"]
    np.testing.assert_array_equal(probabilities[1], [1.0, 0.0])
    np.testing.assert_array_equal(
        np.asarray(payload["cumulative_probabilities"])[:, -1], [1.0, 1.0]
    )
    np.testing.assert_array_equal(
        payload["uniforms"],
        stable_identity_uniforms(lineage_keys, seed=0, salt="brma:clone_spread"),
    )
    for entity in frame.entities:
        pd.testing.assert_frame_equal(frame.table(entity), original[entity])
    np.testing.assert_array_equal(
        frame.weights_for("household").values, original_weights
    )
