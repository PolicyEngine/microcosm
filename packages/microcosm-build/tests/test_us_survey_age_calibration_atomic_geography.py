"""Actual optional age calibration over invented sources and atomic support."""

import hashlib
import json

import numpy as np
import pandas as pd
import test_us_national_age_counts as targets
from test_us_graph_atomic_survey_population import known_atomic_run  # noqa: F401

from microcosm.build.us_runtime import current_survey_geography as observed
from microcosm.build.us_runtime import survey_age_calibration as age
from microcosm.build.us_runtime import survey_population_replay as replay
from microcosm.frame import WeightKind


def test_atomic_age_cold_and_required_replay_retain_raw_budget_and_geography(request):
    case = request.getfixturevalue("known_atomic_run")
    arguments = dict(case.arguments)
    arguments["seed_value"] = arguments.pop("seed")
    runs = tuple(
        age.run_survey_age_calibration(
            **arguments,
            store_root=case.store_root,
            geography_config=case.config,
            target_registry=targets.fixture_registry(),
            epochs=2,
            learning_rate=0.1,
            resume=resume,
        )
        for resume in ("auto", "require")
    )
    cold, warm = runs
    age_ids = {
        age.COUNT_NODE,
        age.transport.BUDGET_NODE,
        age.numerical.CALIBRATION_NODE,
    }
    prefix_ids = set(case.cold.compiled.order)
    assert len(prefix_ids) == 9
    assert cold.manifest.key == warm.manifest.key
    assert cold.budget.payload == warm.budget.payload
    assert all(record.hit for record in warm.manifest.nodes.values())
    assert all(not cold.manifest.node(name).hit for name in age_ids)
    assert all(cold.manifest.node(name).hit for name in prefix_ids)

    geography_columns = tuple(
        column
        for column in case.cold.geography_population.frame.table("household")
        if column not in case.cold.allocated_population.frame.table("household")
    )
    assert set(observed.COLUMNS) < set(geography_columns)
    assert "census_block_geoid" in geography_columns
    raw_identity = age.budgets._population_identity(case.cold.allocated_population)
    for run in runs:
        assert len(run.compiled.order) == len(run.manifest.nodes) == 12
        assert (
            "geography.gate"
            in run.compiled.predecessors[age.numerical.CALIBRATION_NODE]
        )
        assert set(run.compiled.order) == prefix_ids | age_ids
        assert tuple(name for name in run.compiled.order if name in prefix_ids) == (
            case.cold.compiled.order
        )
        for name in prefix_ids:
            assert age._node_identity(run.manifest.node(name)) == age._node_identity(
                case.cold.manifest.node(name)
            )

        # Read the actual issuer entries retained by the completed runner. This
        # adds no synthetic authority and avoids another full source rebuild.
        budget_entry = age.budgets._entry(run.budget, age.budgets.SamplingOriginBudget)
        budget_state = budget_entry[2]
        successor_entry = age.budgets._entry(
            run.successor, age.budgets.SamplingOriginSuccessor
        )
        successor = successor_entry[2]
        raw, previous, current = (
            budget_state.allocated,
            successor.previous,
            successor.current,
        )
        assert successor.budget is run.budget
        assert previous is budget_state.expanded
        assert budget_state.geography_config is case.config
        assert raw.version == age.source_graph.ALLOCATION_NODE
        assert raw.frame.weights_for("household").kind is WeightKind.IMPORTANCE
        assert not set(geography_columns) & set(raw.frame.table("household"))
        replay.same_replayed_population(case.cold.allocated_population, raw)
        replay.same_replayed_population(case.cold.clone_population, previous)
        assert current.version == age.numerical.CALIBRATION_NODE
        assert current.frame.weights_for("household").kind is WeightKind.CALIBRATED
        assert current.frame.n("household") == 2 * raw.frame.n("household")
        age.budgets._same_nonweight(previous.frame, current.frame)
        pd.testing.assert_frame_equal(
            previous.frame.table("household").loc[:, list(geography_columns)],
            current.frame.table("household").loc[:, list(geography_columns)],
            check_exact=True,
        )
        np.testing.assert_array_equal(
            previous.design_weights["household"], current.design_weights["household"]
        )
        replay.same_replayed_frame(
            current.frame, run.manifest.population(age.numerical.CALIBRATION_NODE)
        )
        document = json.loads(run.budget.payload)
        assert (
            document["atomic_geography"]["config_sha256"]
            == hashlib.sha256(case.config.to_bytes()).hexdigest()
        )
        assert (
            document["atomic_geography"]["support_sha256"] == case.config.support_sha256
        )
        assert document["release_eligible"] is run.release_eligible is False
    replay.same_replayed_frame(
        cold.manifest.population(age.numerical.CALIBRATION_NODE),
        warm.manifest.population(age.numerical.CALIBRATION_NODE),
    )
    assert (
        age.budgets._population_identity(case.cold.allocated_population) == raw_identity
    )
