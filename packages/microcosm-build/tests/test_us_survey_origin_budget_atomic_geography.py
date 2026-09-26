"""Budget reconstruction over actual geography/clone runs and invented sources."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from test_us_graph_atomic_survey_population import known_atomic_run  # noqa: F401
from test_us_survey_origin_budget import _assert_final_mutation_refused

from microcosm.build.us_runtime import survey_atomic_geography as reconstruction
from microcosm.build.us_runtime import survey_origin_budget as owner
from microcosm.build.us_runtime import survey_population_replay as replay
from microcosm.frame import Frame


def test_atomic_support_fifo_refuses_before_any_source_borrow(tmp_path, monkeypatch):
    path = tmp_path / "invented-support.fifo"
    os.mkfifo(path)
    config = reconstruction.AtomicSurveyReconstruction(
        support_path=str(path),
        support_sha256="0" * 64,
        source_ids=tuple(
            (name, "invented-" + name) for name in ("district", "population", "puma")
        ),
        seed=0,
    )
    original_open = os.open

    def guarded_open(filename, flags, *args, **kwargs):
        if filename == str(path):
            # Fail promptly on regression, then perform the real descriptor open.
            assert flags & os.O_NONBLOCK
        return original_open(filename, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    with pytest.raises(ValueError, match="ATOMIC_SURVEY_RECONSTRUCTION_SUPPORT_SIZE"):
        reconstruction._read_support(config)


def _arguments(case, *, warm=False):
    run = case.warm if warm else case.cold
    return {
        "preparation": run.preparation,
        "allocated_population": run.allocated_population,
        "clone_population": run.clone_population,
        "geography_config": case.config,
    }


@pytest.fixture(scope="module")
def atomic_budget(request):
    case = request.getfixturevalue("known_atomic_run")
    budget = owner.freeze_survey_origin_budget(**_arguments(case))
    original_config = case.config.to_bytes()
    yield case, budget
    assert case.config.to_bytes() == original_config
    assert Path(case.config.support_path).read_bytes() == case.payload


def test_atomic_budget_binds_complete_geography_on_cold_and_required_replay(
    atomic_budget,
):
    case, budget = atomic_budget
    view = budget.checked_view()
    assert view.allocated_population is case.cold.allocated_population
    assert view.initial_population is case.cold.clone_population
    assert view.allocated_population is not case.cold.geography_population
    binding = view.document["atomic_geography"]
    assert (
        binding["config_sha256"] == hashlib.sha256(case.config.to_bytes()).hexdigest()
    )
    assert binding["support_sha256"] == case.config.support_sha256
    expected = reconstruction.reconstruct_atomic_survey_geography(
        case.cold.preparation, case.cold.allocated_population, case.config
    )
    replay.same_replayed_population(expected.population, case.cold.geography_population)
    expected_receipt = json.loads(expected.receipt)
    # Keep the helper's own physical seal exact; compare independently replayed
    # objects with the maintained named null-storage convention instead.
    for key, population in (
        ("population_sha256", expected.population),
        ("observed_population_sha256", expected.observed_population),
        ("expanded_population_sha256", expected.expanded_population),
    ):
        assert expected_receipt.pop(key) == reconstruction._population_stamp(population)
    assert "preclone_population_semantic_sha256" not in binding
    assert len(binding["postclone_geography_population_semantic_sha256"]) == 64
    assert binding["validation_receipt"] == json.loads(expected.stages[-1].receipt)
    assert (
        binding["reconstruction_semantic_sha256"]
        == hashlib.sha256(owner._json(expected_receipt)).hexdigest()
    )
    assert binding["identity_scope"] == {
        "reconstruction_omitted_fields": [
            "population_sha256",
            "observed_population_sha256",
            "expanded_population_sha256",
        ],
        "population_fields": [
            "frame_sha256",
            "version",
            "owners",
            "weight_kind",
            "mass_ledger",
            "design_weights",
        ],
        "null_backing": "retained_in_in_process_physical_seals_only",
    }
    assert view.document["group_count"] == 6
    assert view.document["release_eligible"] is False
    for run in (case.cold, case.warm):
        view.grouped_bounds.check(
            run.clone_population.frame.weights_for("household").values,
            positive=False,
        )
    assert all(record.hit for record in case.warm.manifest.nodes.values())
    warm = owner.freeze_survey_origin_budget(
        **_arguments(case, warm=True), candidate=budget.payload
    )
    assert warm.payload == budget.payload


@pytest.mark.parametrize(
    "defect,reason",
    (
        ("geography_cell", "SURVEY_POPULATION_REPLAY_STRING_VALUE"),
        ("geography_owner", "SURVEY_POPULATION_REPLAY_POPULATION_CONTEXT"),
        ("clone_discriminator", "SURVEY_POPULATION_REPLAY_NATIVE_BITS"),
    ),
)
def test_atomic_budget_refuses_changed_geography_in_first_clone(
    atomic_budget, defect, reason
):
    case, _budget = atomic_budget
    arguments = _arguments(case)
    changed = reconstruction._copy_population(arguments["clone_population"])
    if defect == "geography_cell":
        changed.frame.table("household").loc[0, "census_block_geoid"] = "0" * 15
    elif defect == "clone_discriminator":
        table = changed.frame.table("household")
        table.loc[0, "household_support_clone_index"] = 1
    else:
        owners = dict(changed.owners)
        owners["household", "census_block_geoid"] = (
            owner.clone.COMBINED_CLONE_CLAIM_NODE
        )
        object.__setattr__(changed, "owners", owners)
    arguments["clone_population"] = changed
    with pytest.raises(ValueError, match=reason):
        owner.freeze_survey_origin_budget(**arguments)


def test_atomic_budget_refuses_preclone_assignment_candidate(atomic_budget):
    case, _budget = atomic_budget
    run = case.cold
    before = run.observed_population.frame
    definition = reconstruction.blocks.assignment_definition(
        identity=(reconstruction.observed.COLUMNS[0],),
        state_column=reconstruction.observed.COLUMNS[1],
        puma_column=reconstruction.observed.COLUMNS[2],
        source_ids=dict(case.config.source_ids),
        seed=case.config.seed,
    )
    supports = {
        reconstruction.blocks.SYSTEM: reconstruction.atomic.decode_atomic_support(
            case.payload
        )
    }
    households = before.table("household")
    assigned = pd.concat(
        [
            households,
            reconstruction.atomic.assign_atomic(households, definition, supports),
        ],
        axis=1,
    )
    tables = {
        entity: before.table(entity).copy(deep=True) for entity in before.entities
    }
    tables["household"] = pd.concat(
        [
            assigned,
            reconstruction.atomic.derive_geography(assigned, definition, supports),
        ],
        axis=1,
    )
    old_geography = Frame(
        tables,
        before.schema,
        {entity: before.weights_for(entity) for entity in before.weighted_entities},
        before.strata,
        metadata=before.metadata,
        mass_log=before.mass_log,
    )
    old_clone = reconstruction.puf_support.clone_us_frame_for_puf_support(
        old_geography, clone_attachment_fraction=1.0, clone_attachment_seed=0
    )
    # Isolate obsolete placement values from unrelated storage differences
    # and retain the actual postclone owners, design anchors and ledger.
    candidate = reconstruction._copy_population(run.clone_population)
    actual_households = candidate.frame.table("household")
    old_households = old_clone.table("household")
    geography_columns = tuple(
        column for column in tables["household"] if column not in households
    )
    assert geography_columns and "census_block_geoid" in geography_columns
    for identity in (
        "household_id",
        reconstruction.observed.COLUMNS[0],
        "household_support_clone_index",
    ):
        assert old_households[identity].tolist() == actual_households[identity].tolist()
    expected_owners = tuple(sorted(candidate.owners.items()))
    old_blocks = old_households["census_block_geoid"].tolist()
    actual_blocks = actual_households["census_block_geoid"].tolist()
    assert all(type(value) is str for value in (*old_blocks, *actual_blocks))
    assert any(a != b for a, b in zip(old_blocks, actual_blocks, strict=True))
    for column in geography_columns:
        reference = actual_households[column]
        actual_households[column] = pd.Series(
            old_households[column].array,
            index=reference.index,
            dtype=reference.dtype,
            name=column,
        )
        assert actual_households[column].dtype == reference.dtype
    assert tuple(sorted(candidate.owners.items())) == expected_owners
    arguments = _arguments(case)
    arguments["clone_population"] = candidate
    with pytest.raises(
        ValueError,
        match="^SURVEY_POPULATION_REPLAY_STRING_VALUE$",
    ):
        owner.freeze_survey_origin_budget(**arguments)


@pytest.mark.parametrize("target", ("recipe", "support"))
def test_atomic_budget_borrow_rechecks_retained_recipe_and_support(
    atomic_budget, target
):
    case, budget = atomic_budget
    config = case.config
    original_seed = config.seed
    support_path = Path(config.support_path)
    try:
        if target == "recipe":
            object.__setattr__(config, "seed", original_seed + 1)
        else:
            support_path.write_bytes(case.payload + b"\n")
        with pytest.raises(ValueError, match="GEOGRAPHY_CONFIG|SUPPORT_CHANGED"):
            budget.checked_view()
    finally:
        object.__setattr__(config, "seed", original_seed)
        if target == "support":
            support_path.write_bytes(case.payload)


def test_atomic_budget_refuses_recipe_mutation_after_final_reconstruction(
    atomic_budget,
):
    case, budget = atomic_budget
    original_seed = case.config.seed

    def mutate():
        object.__setattr__(case.config, "seed", original_seed + 1)

    def restore():
        object.__setattr__(case.config, "seed", original_seed)

    _assert_final_mutation_refused(owner._initial, budget.checked_view, mutate, restore)


def test_atomic_budget_refuses_detached_reconstruction_return_mutation(atomic_budget):
    _case, budget = atomic_budget
    changed = []
    helper = reconstruction.reconstruct_atomic_survey_geography

    def profile(frame, event, result):
        if event == "return" and frame.f_code is helper.__code__ and result is not None:
            changed.append(True)
            result.population.frame.table("household").loc[0, "census_block_geoid"] = (
                "0" * 15
            )

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            owner.SurveyOriginBudgetError, match="GEOGRAPHY_RECONSTRUCTION_SEAL"
        ):
            budget.checked_view()
    finally:
        sys.setprofile(previous)
    assert changed == [True]
