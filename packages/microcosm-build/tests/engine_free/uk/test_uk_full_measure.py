"""Shared full-build measure evaluation retains legacy engine/RNG boundaries."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import full_measure
from microcosm.build.uk_runtime.rowwise_dataset import (
    clone_uk_dataset_with_ladder_geography,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from test_support.microcosm_build.uk_full_population_graph import source_frame
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)


def test_full_measure_reuses_one_resolver(
    monkeypatch,
    tmp_path,
) -> None:
    frame = source_frame()
    constructions = []
    cgt_period_contract = {
        "version": "uk-cgt-measurement-v2",
        "input_period": "2024",
        "calibration_period": 2025,
        "bound_measurements": {
            "cgt_2024_gains": {
                "model_variable": "capital_gains",
                "measurement_period": 2024,
            }
        },
    }

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.simulation = object()
            self.contract_targets = {}

        def receipt(self):
            return {
                "mode": "stub",
                "policyengine_uk_version": "test",
                "cgt_period_contract": cgt_period_contract,
            }

    monkeypatch.setattr(
        full_measure,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {f"{area_type}_metric": np.ones(len(household_ids))},
            index=household_ids,
        ),
    )
    registry = TargetRegistry([], country="uk")
    prepared, restore, national, local_metrics, receipt = (
        full_measure.resolve_uk_full_measures(
            frame,
            registry,
            period=2025,
            scratch_dir=tmp_path / "scratch",
            resolver_factory=StubResolver,
        )
    )

    assert len(constructions) == 1
    # The resolution loop's own receipt rides the measure receipt, one per
    # engine block; its shape is the resolver's, checked by its keys here.
    resolution = receipt.pop("resolution")
    assert [sorted(entry) for entry in resolution] == [
        ["attached", "provider", "rounds"]
    ]
    assert receipt == {
        "mode": "stub",
        "engine_version": "test",
        "households": frame.n("household"),
        "persons": frame.n("person"),
        "benunits": frame.n("benunit"),
        "national_inputs": 0,
        "local_metrics": {"constituency": 1, "la": 1},
        "blocks": 1,
        "target_materialization": {
            "prepared_count": 0,
            "skipped_count": 0,
            "skipped": [],
        },
        "cgt_period_contract": cgt_period_contract,
    }
    assert set(local_metrics) == {"constituency", "la"}
    assert len(national.targets) == 0
    assert restore(prepared).table("household").equals(frame.table("household"))


@pytest.mark.parametrize("second_cgt_period", [2024, 2025, None])
def test_full_measure_resolves_real_per_clone_blocks(
    monkeypatch,
    tmp_path,
    second_cgt_period,
    toy_ladder,
) -> None:
    frame = source_frame()
    ladder, _ = toy_ladder
    clone = clone_uk_dataset_with_ladder_geography(
        frame,
        ladder,
        n_clones=2,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
        source_lineage_modulus=None,
    )
    constructions = []

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.simulation = object()
            self.contract_targets = {}
            self.cgt_period = 2024 if len(constructions) == 1 else second_cgt_period

        def receipt(self):
            receipt = {"mode": "stub", "policyengine_uk_version": "test"}
            if self.cgt_period is not None:
                receipt["cgt_period_contract"] = {
                    "version": "uk-cgt-measurement-v2",
                    "input_period": "2024",
                    "calibration_period": 2025,
                    "bound_measurements": {
                        "cgt_2024_gains": {
                            "model_variable": "capital_gains",
                            "measurement_period": self.cgt_period,
                        }
                    },
                }
            return receipt

    monkeypatch.setattr(
        full_measure,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {f"{area_type}_metric": np.arange(len(household_ids), dtype=float)},
            index=household_ids,
        ),
    )

    def resolve():
        return full_measure.resolve_uk_full_measures(
            clone.frame,
            TargetRegistry([], country="uk"),
            period=2025,
            scratch_dir=tmp_path / "block-scratch",
            resolver_factory=StubResolver,
            blocks=2,
        )

    if second_cgt_period != 2024:
        with pytest.raises(RuntimeError, match="CGT period contract is inconsistent"):
            resolve()
        return

    prepared, restore, _, metrics, receipt = resolve()

    assert len(constructions) == 2
    assert [len(call["frame"].table("household")) for call in constructions] == [
        frame.n("household"),
        frame.n("household"),
    ]
    assert receipt["blocks"] == 2
    assert receipt["cgt_period_contract"]["bound_measurements"] == {
        "cgt_2024_gains": {
            "model_variable": "capital_gains",
            "measurement_period": 2024,
        }
    }
    assert receipt["deviation"] == "per_clone_block_engine_resolution"
    sensitivity = receipt["block_sensitivity"]
    assert (
        "ons/corporate_land_value"
        in sensitivity["known_population_normalised_measures"]
    )
    assert set(sensitivity["present_in_this_run"]) <= set(
        sensitivity["known_population_normalised_measures"]
    )
    assert "not evidence for adjudication" in sensitivity["caveat"]
    assert (
        metrics["constituency"].index.tolist()
        == clone.frame.table("household")["household_id"].tolist()
    )
    assert restore(prepared).table("household").equals(clone.frame.table("household"))


@pytest.mark.parametrize("blocks", [1, 2])
def test_prepared_measures_preserve_ids_and_remove_duplicate_scratch_inputs(
    monkeypatch, tmp_path, toy_ladder, blocks
):
    frame = source_frame()
    if blocks == 2:
        frame = clone_uk_dataset_with_ladder_geography(
            frame,
            toy_ladder[0],
            n_clones=2,
            seed=7,
            source_year=2023,
            expected_constituency_vintage="2024_pcon",
        ).frame
    registry = TargetRegistry(
        [
            TargetSpec(
                name="national",
                entity="household",
                measure="prepared_count",
                value=33.0,
                period=2025,
                family="fixture",
                source="fixture",
            )
        ],
        country="uk",
    )

    class Resolver:
        def __init__(self, *, frame, **kwargs):
            self.frame = frame
            self.simulation = object()

        def receipt(self):
            return {"mode": "stub", "policyengine_uk_version": "test"}

    monkeypatch.setattr(
        full_measure,
        "resolve_target_measures",
        lambda _factory, _registry, provider, **kwargs: SimpleNamespace(
            receipt={"attached": {}, "provider": {}, "rounds": []},
            measure_inputs={
                ("person", "region"): np.zeros(provider.frame.n("person")),
                ("household", "raw_engine_input"): provider.frame.table("household")[
                    "household_id"
                ].to_numpy(),
                ("household", "ons/corporate_land_value"): np.ones(
                    provider.frame.n("household")
                ),
            },
        ),
    )

    def materialize(adapter, _registry, **kwargs):
        # The duplicate scratch region must be usable while materializing,
        # then removed before assembling the flattened prepared Frame.
        assert "region" in adapter.tables["person"]
        table = adapter.tables["household"]
        table["prepared_count"] = table["raw_engine_input"].to_numpy()
        return SimpleNamespace(
            skipped=(),
            report=lambda: {"prepared_count": 0, "skipped_count": 0, "skipped": []},
        )

    monkeypatch.setattr(full_measure, "materialize_uk_ledger_targets", materialize)
    prepared, restore, national, metrics, receipt = (
        full_measure.resolve_uk_full_measures(
            frame,
            registry,
            period=2025,
            scratch_dir=tmp_path / "scratch",
            resolver_factory=Resolver,
            blocks=blocks,
            local_grains=(),
        )
    )
    assert "region" not in prepared.table("person")
    assert "raw_engine_input" not in prepared.table("household")
    np.testing.assert_array_equal(
        prepared.table("household")["prepared_count"],
        frame.table("household")["household_id"],
    )
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            restore(prepared).table(entity), frame.table(entity)
        )
    assert len(national.targets) == 1
    assert metrics == {}
    assert receipt["national_inputs"] == 3
    if blocks == 2:
        assert receipt["block_sensitivity"]["present_in_this_run"] == [
            "ons/corporate_land_value"
        ]


def test_scratch_paths_are_recorded_relative_to_the_scratch_root(monkeypatch, tmp_path):
    """A scratch-mode resolver names the file it simulated from, under the
    per-run temporary directory; the measure receipt records that path
    relative to the scratch root, so two identical resolutions in different
    scratch directories yield the same receipt (the measure and national
    problem nodes are declared deterministic; #1057 round 1)."""
    import json

    frame = source_frame()

    class StubResolver:
        def __init__(self, **kwargs):
            self.scratch_dir = kwargs["scratch_dir"]
            self.simulation = object()
            self.contract_targets = {}

        def receipt(self):
            return {
                "mode": "scratch_frame_export",
                "source_path": str(self.scratch_dir / "simulation-input.h5"),
                "policyengine_uk_version": "test",
            }

    monkeypatch.setattr(
        full_measure,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {f"{area_type}_metric": np.ones(len(household_ids))},
            index=household_ids,
        ),
    )
    registry = TargetRegistry([], country="uk")
    receipts = []
    for name in ("first", "second"):
        _, _, _, _, receipt = full_measure.resolve_uk_full_measures(
            frame,
            registry,
            period=2025,
            scratch_dir=tmp_path / name,
            resolver_factory=StubResolver,
        )
        receipts.append(receipt)
    assert receipts[0] == receipts[1]
    provider = receipts[0]["resolution"][0]["provider"]
    assert provider["source_path"] == "simulation-input.h5"
    assert str(tmp_path) not in json.dumps(receipts[0])


@pytest.mark.parametrize("blocks", [1, 2])
def test_each_engine_block_is_collected_before_the_next_loads(
    monkeypatch, tmp_path, toy_ladder, blocks
):
    """A policyengine simulation is a large cyclic object graph that `del`
    alone does not reclaim; the first K=25 build (2026-10-05) retained all 25
    per-clone engines until the machine killed it. The loop collects after
    every block, including a single-block run."""
    frame = source_frame()
    if blocks == 2:
        frame = clone_uk_dataset_with_ladder_geography(
            frame,
            toy_ladder[0],
            n_clones=2,
            seed=7,
            source_year=2023,
            expected_constituency_vintage="2024_pcon",
        ).frame
    registry = TargetRegistry([], country="uk")

    class Resolver:
        def __init__(self, *, frame, **kwargs):
            self.frame = frame
            self.simulation = object()

        def receipt(self):
            return {"mode": "stub", "policyengine_uk_version": "test"}

    monkeypatch.setattr(
        full_measure,
        "resolve_target_measures",
        lambda _factory, _registry, provider, **kwargs: SimpleNamespace(
            receipt={"attached": {}, "provider": {}, "rounds": []},
            measure_inputs={},
        ),
    )
    collections = []
    monkeypatch.setattr(
        full_measure, "gc", SimpleNamespace(collect=lambda: collections.append(1) or 0)
    )
    full_measure.resolve_uk_full_measures(
        frame,
        registry,
        period=2025,
        scratch_dir=tmp_path / "scratch",
        resolver_factory=Resolver,
        blocks=blocks,
        local_grains=(),
    )
    assert len(collections) == blocks


def _national_problem_fixture(monkeypatch, toy_ladder):
    """A two-clone frame, one national target materialized from an injected
    engine input, and the stubs the per-block path needs (the measure
    inputs carry a person-level and two household-level columns, exactly as
    the prepared-measures test above)."""
    frame = clone_uk_dataset_with_ladder_geography(
        source_frame(),
        toy_ladder[0],
        n_clones=2,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
    ).frame
    registry = TargetRegistry(
        [
            TargetSpec(
                name="national",
                entity="household",
                measure="prepared_count",
                value=33.0,
                period=2025,
                family="fixture",
                source="fixture",
            )
        ],
        country="uk",
    )

    class Resolver:
        def __init__(self, *, frame, **kwargs):
            self.frame = frame
            self.simulation = object()

        def receipt(self):
            return {"mode": "stub", "policyengine_uk_version": "test"}

    monkeypatch.setattr(
        full_measure,
        "resolve_target_measures",
        lambda _factory, _registry, provider, **kwargs: SimpleNamespace(
            receipt={"attached": {}, "provider": {}, "rounds": []},
            measure_inputs={
                ("person", "region"): np.zeros(provider.frame.n("person")),
                ("household", "raw_engine_input"): provider.frame.table("household")[
                    "household_id"
                ].to_numpy(),
                ("household", "ons/corporate_land_value"): np.ones(
                    provider.frame.n("household")
                ),
            },
        ),
    )

    def materialize(adapter, _registry, **kwargs):
        assert "region" in adapter.tables["person"]
        table = adapter.tables["household"]
        table["prepared_count"] = table["raw_engine_input"].to_numpy()
        return SimpleNamespace(
            skipped=(),
            report=lambda: {"prepared_count": 1, "skipped_count": 0, "skipped": []},
        )

    monkeypatch.setattr(full_measure, "materialize_uk_ledger_targets", materialize)
    return frame, registry, Resolver


def test_national_problem_is_compiled_per_block_in_the_pools_household_order(
    monkeypatch, tmp_path, toy_ladder
):
    """The dense kernel's path: each clone block materializes and compiles its
    own columns; the stitched problem equals the whole-pool problem, column
    for column, and never materializes the pool (microcosm#932 follow-up)."""
    frame, registry, resolver_factory = _national_problem_fixture(
        monkeypatch, toy_ladder
    )
    results = {}
    for blocks in (1, 2):
        problem, rows, metrics, receipt = full_measure.resolve_uk_full_national_problem(
            frame,
            registry,
            period=2025,
            scratch_dir=tmp_path / f"scratch-{blocks}",
            resolver_factory=resolver_factory,
            blocks=blocks,
            local_grains=(),
        )
        results[blocks] = problem
        ids = frame.table("household")["household_id"].to_numpy()
        assert problem.matrix.shape == (1, len(ids))
        # Each household's contribution sits in its own column, in frame order.
        np.testing.assert_array_equal(problem.matrix.toarray()[0], ids)
        np.testing.assert_array_equal(problem.target_vector, [33.0])
        assert problem.weight_entity == "household"
        np.testing.assert_array_equal(
            problem.initial_weights.values, frame.resolve_weights("household").values
        )
        assert problem.skipped == ()
        assert len(rows.targets) == 1
        assert metrics == {}
        assert receipt["blocks"] == blocks
        assert receipt["national_materialization"] == "per_engine_block"
        assert receipt["target_materialization"] == {
            "prepared_count": 1,
            "skipped_count": 0,
            "skipped": [],
        }
        assert receipt["national_inputs"] == 3
        if blocks == 2:
            assert receipt["deviation"] == "per_clone_block_engine_resolution"
    one, two = results[1], results[2]
    assert (one.matrix != two.matrix).nnz == 0
    assert one.names == two.names
    # The pool itself was left untouched by both paths.
    assert "prepared_count" not in frame.table("household")


def test_national_problem_is_none_without_national_targets(
    monkeypatch, tmp_path, toy_ladder
):
    frame, _registry, resolver_factory = _national_problem_fixture(
        monkeypatch, toy_ladder
    )
    monkeypatch.setattr(
        full_measure,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {f"{area_type}_metric": np.ones(len(household_ids))},
            index=household_ids,
        ),
    )
    problem, rows, metrics, receipt = full_measure.resolve_uk_full_national_problem(
        frame,
        TargetRegistry([], country="uk"),
        period=2025,
        scratch_dir=tmp_path / "scratch",
        resolver_factory=resolver_factory,
        blocks=2,
        local_grains=("constituency",),
    )
    assert problem is None
    assert len(rows.targets) == 0
    assert (
        metrics["constituency"].index.tolist()
        == frame.table("household")["household_id"].tolist()
    )
    assert receipt["national_materialization"] == "per_engine_block"
