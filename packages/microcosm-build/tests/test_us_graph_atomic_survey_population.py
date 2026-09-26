"""Actual survey/geography/clone execution over pinned invented originals.

Normalized invented support proves no publisher provenance or native acceptance.
Invented registry pins bind source fixtures; real issuers and kernels execute.
"""

import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_current_asec_demographics import _demographic_arguments

from microcosm.build import atomic_geography as atomic
from microcosm.build.us_runtime import atomic_block_support as blocks
from microcosm.build.us_runtime import current_survey_geography as observed
from microcosm.build.us_runtime import graph_atomic_survey_population as runner
from microcosm.build.us_runtime import graph_combined_clone as clone
from microcosm.build.us_runtime import graph_current_survey_geography as projection
from microcosm.build.us_runtime import graph_survey_population as survey
from microcosm.build.us_runtime import puf_support
from microcosm.build.us_runtime import survey_atomic_geography as reconstruction
from microcosm.build.us_runtime import survey_population_preparation as source
from microcosm.build.us_runtime import survey_population_replay as replay
from microcosm.frame import WeightKind


def _support_payload(*, first_population=2):
    areas = tuple(
        int(value)
        for value in (
            "060010201001000",
            "060010201001001",
            "060010202001000",
            "360010001001000",
        )
    )
    source_ids = {
        name: "invented-" + name for name in ("district", "population", "puma")
    }
    payload = blocks.assemble_atomic_block_support(
        block_population=dict(zip(areas, (first_population, 6, 3, 5), strict=True)),
        cd_by_block=dict(zip(areas, (601, 602, 603, 3601), strict=True)),
        puma_by_tract={
            areas[0] // 10000: 612345,
            areas[2] // 10000: 699999,
            areas[3] // 10000: 3600001,
        },
        source_ids=source_ids,
    )
    return payload, source_ids


@pytest.fixture(scope="module")
def known_atomic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("atomic-survey-population")
    with pytest.MonkeyPatch.context() as patch:
        arguments = _demographic_arguments(root, patch, unknown=False)
        payload, source_ids = _support_payload()
        support_path = root / "invented-block-support.npz"
        support_path.write_bytes(payload)
        config = reconstruction.AtomicSurveyReconstruction(
            support_path=str(support_path),
            support_sha256=hashlib.sha256(payload).hexdigest(),
            source_ids=tuple(sorted(source_ids.items())),
            seed=17,
        )
        store_root = root / "store"
        runs = tuple(
            runner.run_atomic_survey_population(
                **arguments,
                store_root=store_root,
                geography_config=config,
                resume=resume,
                return_values=True,
            )
            for resume in ("auto", "require")
        )
        yield SimpleNamespace(
            arguments=arguments,
            payload=payload,
            config=config,
            store_root=store_root,
            cold=runs[0],
            warm=runs[1],
        )
        for run in runs:
            source.verify_survey_population_preparation(run.preparation)


def _assert_complete_clone(before, after):
    expected = puf_support.clone_us_frame_for_puf_support(
        before.frame, clone_attachment_fraction=1.0, clone_attachment_seed=0
    )
    replay.same_replayed_frame(expected, after.frame)
    # Independently pair both roles by assembly source ID, so inheritance is
    # checked independently of the clone operator's row ordering/remapping.
    for entity in before.frame.entities:
        source_id = puf_support.support_source_id_column(entity)
        clone_index = puf_support.support_clone_index_column(entity)
        original = before.frame.table(entity).set_index(
            source_id, verify_integrity=True
        )
        expanded = after.frame.table(entity)
        assert len(expanded) == 2 * len(original)
        assert set(expanded[clone_index]) == {0, 1}
        remapped = {before.frame.schema.entity_id_column(entity), clone_index}
        if entity == before.frame.schema.person_entity:
            remapped.update(
                before.frame.schema.membership_column(group)
                for group in before.frame.schema.group_entities
            )
        inherited = [column for column in original if column not in remapped]
        for role in (0, 1):
            arm = expanded.loc[expanded[clone_index].eq(role)].set_index(
                source_id, verify_integrity=True
            )
            pd.testing.assert_index_equal(arm.index, original.index, exact=True)
            pd.testing.assert_frame_equal(
                arm.loc[original.index, inherited],
                original.loc[:, inherited],
                check_exact=True,
            )
    assert after.version == clone.COMBINED_CLONE_NODE
    assert after.mass_ledger[:-1] == before.mass_ledger
    assert (
        after.weight_kind == before.weight_kind == {"household": WeightKind.IMPORTANCE}
    )
    np.testing.assert_array_equal(
        after.frame.weights_for("household").values,
        np.tile(before.frame.weights_for("household").values / 2, 2),
    )
    assert set(after.design_weights) == set(before.design_weights) == {"household"}
    np.testing.assert_array_equal(
        after.design_weights["household"],
        np.tile(before.design_weights["household"], 2),
    )
    assert dict(after.owners) == {
        (entity, column): (
            clone.COMBINED_CLONE_CLAIM_NODE
            if column == puf_support.support_clone_index_column(entity)
            else clone.COMBINED_CLONE_NODE
        )
        for entity in after.frame.entities
        for column in after.frame.table(entity)
    }


def test_cold_and_required_replay_assign_after_complete_clone(known_atomic_run):
    case = known_atomic_run
    assert case.cold.manifest.key == case.warm.manifest.key
    assert all(record.hit for record in case.warm.manifest.nodes.values())
    assert all(
        not record.hit
        for name, record in case.cold.manifest.nodes.items()
        if name not in {survey.CREATE_NODE, survey.ALLOCATION_NODE}
    )
    for field in (
        "allocated_population",
        "observed_population",
        "expanded_population",
        "geography_population",
        "clone_population",
    ):
        replay.same_replayed_population(
            getattr(case.cold, field), getattr(case.warm, field)
        )
    support = atomic.decode_atomic_support(case.payload)
    for run in (case.cold, case.warm):
        raw, observed_population = run.allocated_population, run.observed_population
        expanded, geography = run.expanded_population, run.geography_population
        assert run.clone_population is geography
        assert raw.version == observed_population.version == survey.ALLOCATION_NODE
        assert expanded.version == geography.version == clone.COMBINED_CLONE_NODE
        assert len(run.compiled.order) == 9
        stages = (
            survey.ALLOCATION_NODE,
            projection.NODE,
            clone.COMBINED_CLONE_NODE,
            clone.COMBINED_CLONE_CLAIM_NODE,
            "geography.assign",
            "geography.derive",
            "geography.gate",
        )
        positions = tuple(run.compiled.order.index(name) for name in stages)
        assert positions == tuple(sorted(positions))
        assert (
            clone.COMBINED_CLONE_CLAIM_NODE
            in run.compiled.predecessors["geography.assign"]
        )
        rebuilt = reconstruction.reconstruct_atomic_survey_geography(
            run.preparation, raw, case.config
        )
        for expected, actual in (
            (rebuilt.observed_population, observed_population),
            (rebuilt.expanded_population, expanded),
            (rebuilt.population, geography),
        ):
            replay.same_replayed_population(expected, actual)
        receipt, definition = (
            json.loads(rebuilt.receipt),
            json.loads(rebuilt.definition),
        )
        assert receipt["protocol"] == "microcosm.us.atomic-survey-reconstruction.v2"
        assert receipt["assignment_identity"] == list(
            reconstruction.composition.ASSIGNMENT_IDENTITY
        )
        assert rebuilt.support_payload == case.payload
        assert receipt["support_sha256"] == case.config.support_sha256
        for name in (
            "publisher_provenance_established",
            "source_admission_issued",
            "population_admission_issued",
            "release_eligible",
        ):
            assert receipt[name] is False
        assigned = tuple(definition["outputs"].values())
        derived = tuple(layer["output"] for layer in definition["systems"][0]["layers"])
        assert not set((*observed.COLUMNS, *assigned, *derived)) & set(
            raw.frame.table("household")
        )
        assert not set((*assigned, *derived)) & set(expanded.frame.table("household"))
        _assert_complete_clone(observed_population, expanded)
        # Both ordinary projection steps preserve every incumbent cell, entity,
        # membership, axis, weight, design anchor and context field exactly.
        for before, after, added in (
            (raw, observed_population, observed.COLUMNS),
            (expanded, geography, (*assigned, *derived)),
        ):
            assert set(after.frame.table("household")) - set(
                before.frame.table("household")
            ) == set(added)
            for entity in before.frame.entities:
                pd.testing.assert_frame_equal(
                    after.frame.table(entity).loc[
                        :, before.frame.table(entity).columns
                    ],
                    before.frame.table(entity),
                    check_exact=True,
                )
            assert after.frame.schema == before.frame.schema
            assert after.frame.entities == before.frame.entities
            assert after.frame.links == before.frame.links == ()
            assert after.frame.metadata == before.frame.metadata
            assert after.frame.mass_log == before.frame.mass_log
            assert after.mass_ledger == before.mass_ledger
            assert after.weight_kind == before.weight_kind
            assert after.frame.weighted_entities == before.frame.weighted_entities
            pd.testing.assert_series_equal(
                after.frame.strata, before.frame.strata, check_exact=True
            )
            for entity in before.design_weights:
                np.testing.assert_array_equal(
                    after.design_weights[entity], before.design_weights[entity]
                )
                np.testing.assert_array_equal(
                    after.frame.weights_for(entity).values,
                    before.frame.weights_for(entity).values,
                )
        assert dict(observed_population.owners) == {
            **raw.owners,
            **{("household", name): projection.NODE for name in observed.COLUMNS},
        }
        assert dict(geography.owners) == {
            **expanded.owners,
            **{("household", name): "geography.assign" for name in assigned},
            **{("household", name): "geography.derive" for name in derived},
        }
        replay.same_replayed_frame(
            observed_population.frame, run.manifest.population(survey.ALLOCATION_NODE)
        )
        replay.same_replayed_frame(
            geography.frame, run.manifest.population(clone.COMBINED_CLONE_NODE)
        )
        households = geography.frame.table("household")
        assert len(households) == 12
        assert not households.duplicated(
            list(reconstruction.composition.ASSIGNMENT_IDENTITY)
        ).any()
        keys = households[observed.COLUMNS[0]].map(json.loads)
        acs = keys.map(lambda key: key[0] == "acs")
        assert acs.sum() == 8
        assert households.loc[acs, "survey_observed_puma"].eq("0612345").all()
        assert households.loc[acs, "assigned_puma_geoid"].eq("0612345").all()
        assert households.loc[~acs, "survey_observed_puma"].isna().all()
        for native, state in (("00007", "06"), ("00008", "36")):
            selected = keys.map(
                lambda key, native=native: key[0] == "asec" and key[-1] == native
            )
            assert selected.sum() == 2
            assert households.loc[selected, "survey_observed_state"].tolist() == [
                state,
                state,
            ]
            assert households.loc[selected, "assigned_state_fips"].tolist() == [
                state,
                state,
            ]
        assert households.loc[acs, "assigned_state_fips"].eq("06").all()
        assert set(households.census_block_geoid) <= set(support.arrays["area"])
        atomic.validate_geography(households, definition, {blocks.SYSTEM: support})
        gate = run.manifest.node("geography.gate")
        assert gate.receipt["outcome"] == "pass"
        assert (
            run.store.load_bytes(gate.opaque_artifacts["validation"])
            == rebuilt.stages[-1].receipt
        )


def test_changed_normalized_support_is_refused_by_exact_digest(
    known_atomic_run, tmp_path
):
    case = known_atomic_run
    path = tmp_path / "changed-invented-support.npz"
    changed, _ = _support_payload(first_population=3)
    assert changed != case.payload
    atomic.decode_atomic_support(changed)
    path.write_bytes(changed)
    config = replace(case.config, support_path=str(path))
    with pytest.raises(
        ValueError, match="ATOMIC_SURVEY_RECONSTRUCTION_SUPPORT_CHANGED"
    ):
        reconstruction.reconstruct_atomic_survey_geography(
            case.cold.preparation, case.cold.allocated_population, config
        )


def test_enriched_population_cannot_substitute_for_raw_allocation(known_atomic_run):
    run = known_atomic_run.cold
    with pytest.raises(ValueError, match="MATERIALIZED_FRAME_VALUES"):
        reconstruction.reconstruct_atomic_survey_geography(
            run.preparation, run.geography_population, known_atomic_run.config
        )


@pytest.mark.parametrize(
    "change,reason",
    (
        ("observed_cell", "SURVEY_POPULATION_REPLAY_"),
        ("observed_snapshot", "SURVEY_POPULATION_REPLAY_"),
        ("expanded_snapshot", "SURVEY_POPULATION_REPLAY_"),
        ("source_container", "ATOMIC_FINAL_RECONSTRUCTION"),
    ),
)
def test_late_returned_population_or_sources_mutation_refuses(
    known_atomic_run, change, reason
):
    case, fired = known_atomic_run, []

    def trace(frame, event, arg):
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code
            is reconstruction.reconstruct_atomic_survey_geography.__code__
            and caller is not None
            and caller.f_code is runner.run_atomic_survey_population.__code__
            and "result" in caller.f_locals
            and not fired
        ):
            fired.append(True)
            result = caller.f_locals["result"]
            if change == "observed_cell":
                households = result.geography_population.frame.table("household")
                households.loc[households.index[0], "survey_observed_state"] = "99"
            elif change in {"observed_snapshot", "expanded_snapshot"}:
                value = (
                    result.observed_population
                    if change == "observed_snapshot"
                    else result.expanded_population
                )
                households = value.frame.table("household")
                households.loc[households.index[0], "survey_observed_state"] = "99"
            else:
                result.sources[blocks.SOURCE] = (
                    str(result.sources[blocks.SOURCE]) + ".changed"
                )

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(ValueError, match=reason):
            runner.run_atomic_survey_population(
                **case.arguments,
                store_root=case.store_root,
                geography_config=case.config,
                resume="require",
                return_values=True,
            )
    finally:
        sys.setprofile(previous)
    assert fired == [True]
