"""Shared full-build measure evaluation retains legacy engine/RNG boundaries."""

import copy
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import full_measure
from microcosm.build.uk_runtime.rowwise_dataset import (
    clone_uk_dataset_with_ladder_geography,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame.adapters.engine_population import (
    MEASURE_PASS,
    EngineBlock,
    PoolShare,
    PopulationVerdict,
    Violation,
)
from microcosm.frame.adapters.policyengine_uk_population import UK_ENGINE_POPULATION
from test_support.microcosm_build.uk_full_population_graph import source_frame
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)
from test_support.microcosm_frame.engine_population_fakes import (
    BLOCKS,
    GROWTH,
    KEY_A,
    D,
    P,
    block_shares,
    fake_block,
    fake_declaration,
    install_fake_engine,
    numbers_are_years,
)


def _clean_block_receipt(shares, *, periods=("2024", "2025")):
    """What a UKMeasureResolver records for an engine block whose correction
    and guard all held (microcosm#931): the shares it applied, both checks,
    one growth factor per key and the weights, siblings on the block's keys,
    and a clean measure-pass verdict."""

    corrections = {formula: share.factor() for formula, share in sorted(shares.items())}
    keys = sorted({share.key_variable for share in shares.values()})
    verdict = PopulationVerdict(
        pass_name="measure",
        violations=(),
        truncated=False,
        classes={
            "corrected": tuple(sorted(UK_ENGINE_POPULATION.corrected)),
            "stored": tuple(sorted(UK_ENGINE_POPULATION.stored)),
            "zero_at_policy": ("carbon_tax",),
            "inert": (),
        },
        window_first_year=UK_ENGINE_POPULATION.window.first_year,
        declaration_sha256=UK_ENGINE_POPULATION.sha256,
        engine="policyengine-uk",
        engine_version="test",
    )
    return {
        "engine_pool_shares": {
            "periods": list(periods),
            "dataset_year": int(periods[0]),
            "formulas": {f: s.key_variable for f, s in sorted(shares.items())},
            "corrections": corrections,
            "declaration_valid": True,
            "key_sums_verified": True,
            "largest_key_sum_gap": 1e-8,
            "uniform_growth_verified": True,
            "growth_factors": {
                period: {
                    **{key: 1.03 for key in keys},
                    UK_ENGINE_POPULATION.weight_variable: 1.004,
                }
                for period in periods[1:]
            },
            "applied": True,
            "siblings_match": True,
            "refusals": [],
        },
        "engine_population_reach": verdict.as_receipt(),
        "annual_income_measures": {
            "other_spells_engine_factors": {"employment_income_before_lsr": 1.04}
        },
    }


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
    # One block over the pool is never corrected or judged (microcosm#931).
    assert "engine_pool_shares" not in constructions[0]
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
    # The weight-share allocation keys ride into every clone, so the
    # representation can verify them (microcosm#1115 review).
    from microcosm.build.uk_runtime.full_measure import (
        UK_WEIGHT_SHARE_FORMULA_INPUTS,
    )
    from microcosm.frame import Frame

    original = source_frame()
    tables = {e: original.table(e).copy() for e in original.entities}
    for inputs in UK_WEIGHT_SHARE_FORMULA_INPUTS.values():
        for column in inputs:
            tables["household"][column] = (
                tables["household"]["household_id"].astype("float64") * 10.0
            )
    frame = Frame(
        tables,
        original.schema,
        {"household": original.weights_for("household")},
        original.strata,
        mass_log=original.mass_log,
        metadata=original.metadata,
    )
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
    closed = []

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.simulation = object()
            self.contract_targets = {}
            self.cgt_period = 2024 if len(constructions) == 1 else second_cgt_period
            self.shares = kwargs.get("engine_pool_shares")

        def close_engine_block(self):
            closed.append(len(constructions))

        def receipt(self):
            receipt = {"mode": "stub", "policyengine_uk_version": "test"}
            if self.shares is not None:
                receipt.update(_clean_block_receipt(self.shares))
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
    # Each block carries half the pool, so its engine weights are scaled by two.
    assert [call["engine_weight_scale"] for call in constructions] == pytest.approx(
        [2.0, 2.0]
    )
    # Each block's resolver gets the pool shares of its share formulas: the
    # clones are identical copies, so c_b = f_b * S_b / S is 1, and each share
    # carries the block's engine weight scale (microcosm#931). Every block is
    # judged once it has computed everything.
    for call in constructions:
        shares = call["engine_pool_shares"]
        assert set(shares) == set(UK_ENGINE_POPULATION.corrected)
        for formula, share in shares.items():
            assert isinstance(share, PoolShare)
            assert share.key_variable == UK_ENGINE_POPULATION.corrected[formula]
            assert share.weight_scale == call["engine_weight_scale"]
            assert share.factor() == pytest.approx(1.0, rel=1e-12)
    assert closed == [1, 2]
    representation = receipt["engine_population_representation"]
    assert representation["mode"] == "block_weights_scaled_to_pool"
    assert representation["exact"] is True
    assert representation["exact_basis"] == full_measure.UK_ENGINE_BLOCK_EXACT_BASIS
    assert representation["exact_scope"] == full_measure.UK_ENGINE_BLOCK_EXACT_SCOPE
    assert representation["checks"]["identical_source_weight_multisets"] is True
    assert representation["share_correction"]["applied_in_every_block"] is True
    assert representation["population_reach"]["clean_in_every_block"] is True
    assert receipt["block_sensitivity"]["exact"] is True
    assert receipt["block_sensitivity"]["caveat"] is None
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
    # The engine variables the two share-formula land rows read, matched
    # against the measure inputs (keyed by engine variable), not the rows'
    # metric names.
    assert sensitivity["known_population_normalised_measures"] == [
        "corporate_land_value",
        "land_value",
    ]
    assert sensitivity["present_in_this_run"] == []
    assert sensitivity["corrected_formulas"] == dict(UK_ENGINE_POPULATION.corrected)
    # Pool-scaled weights and corrected shares: the known formulas see the
    # pool's denominator, so the receipt names the mitigation and no caveat.
    assert "scaled by pool mass / block mass" in sensitivity["mitigation"]
    assert sensitivity["exact"] is True
    assert sensitivity["caveat"] is None
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
                ("household", "corporate_land_value"): np.ones(
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
        # A measure input on the engine variable a land row binds is seen
        # (microcosm#931: the record compared metric names with engine
        # variable names, so it was always empty).
        assert receipt["block_sensitivity"]["present_in_this_run"] == [
            "corporate_land_value"
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


def _representation_block(
    source_ids, weights, persons=3, *, identity=True, key_values=None
):
    """A block whose allocation keys all equal ``key_values`` (default 1 per row)."""
    from microcosm.build.uk_runtime.full_measure import (
        UK_WEIGHT_SHARE_FORMULA_INPUTS,
    )

    n = len(weights)
    columns = {"household_id": np.arange(n)}
    if identity:
        columns["household_source_id"] = np.asarray(source_ids)
    values = np.ones(n) if key_values is None else np.asarray(key_values, dtype=float)
    for inputs in UK_WEIGHT_SHARE_FORMULA_INPUTS.values():
        for column in inputs:
            columns[column] = values
    household = pd.DataFrame(columns)
    person = pd.DataFrame(index=range(persons))
    values = np.asarray(weights, dtype=float)
    return SimpleNamespace(
        table=lambda name: household if name == "household" else person,
        weights_for=lambda entity: SimpleNamespace(
            values=values, total=float(values.sum())
        ),
    )


def test_engine_population_representation_records_the_blocks_before_they_run():
    """Before the blocks run, the representation records each block's factor
    and allocation-key totals and the identical-copies checks, as diagnostics;
    whether the blocks stand for the pool is decided after they ran, so exact
    is provisional (microcosm#931)."""
    pool = SimpleNamespace(weights_for=lambda entity: SimpleNamespace(total=6.0))
    copies = [
        (0, _representation_block([10, 11, 12], [0.5, 1.0, 1.5])),
        (1, _representation_block([12, 10, 11], [1.5, 0.5, 1.0])),
    ]
    record = full_measure._engine_population_representation(pool, copies)
    assert record["exact"] is False
    assert record["exact_basis"] == "pool_share_injection_with_population_guard"
    assert record["mode"] == "block_weights_scaled_to_pool"
    assert record["factor_by_block"] == {
        "0": pytest.approx(2.0),
        "1": pytest.approx(2.0),
    }
    assert record["checks"]["max_abs_mass_share_deviation"] == pytest.approx(0.0)
    assert record["checks"]["identical_source_weight_multisets"] is True
    assert record["checks"]["weight_share_inputs_match"] is True

    skewed = [
        (0, _representation_block([10, 11, 12], [0.5, 1.0, 1.5])),
        (1, _representation_block([10, 11, 12], [0.5, 1.0, 2.0])),
    ]
    pool = SimpleNamespace(weights_for=lambda entity: SimpleNamespace(total=6.5))
    approximate = full_measure._engine_population_representation(pool, skewed)
    assert approximate["checks"]["identical_source_weight_multisets"] is False
    assert approximate["factor_by_block"]["0"] == pytest.approx(6.5 / 3.0)
    assert approximate["factor_by_block"]["1"] == pytest.approx(6.5 / 3.5)

    single = full_measure._engine_population_representation(pool, [(None, object())])
    assert single == {"mode": "single_block", "exact": True, "blocks": 1}
    assert full_measure._block_pool_shares(single) is None
    assert full_measure._finalize_representation(single, [{}], shares_built=False) == {
        "mode": "single_block",
        "exact": True,
        "blocks": 1,
    }

    pool = SimpleNamespace(weights_for=lambda entity: SimpleNamespace(total=6.0))
    nameless = [
        (0, _representation_block([10, 11, 12], [0.5, 1.0, 1.5], identity=False)),
        (1, _representation_block([12, 10, 11], [1.5, 0.5, 1.0], identity=False)),
    ]
    anonymous = full_measure._engine_population_representation(pool, nameless)
    assert anonymous["checks"]["identity_key"] is None
    assert anonymous["checks"]["source_identity_present"] is False


def test_identical_copies_take_a_unit_correction_and_differing_keys_split_the_pool():
    pool = SimpleNamespace(weights_for=lambda entity: SimpleNamespace(total=6.0))
    copies = [
        (0, _representation_block([10, 11, 12], [0.5, 1.0, 1.5], key_values=[1, 2, 3])),
        (1, _representation_block([12, 10, 11], [1.5, 0.5, 1.0], key_values=[3, 1, 2])),
    ]
    shares = full_measure._block_pool_shares(
        full_measure._engine_population_representation(pool, copies)
    )
    assert set(shares) == {"0", "1"}
    for block in shares.values():
        assert set(block) == set(UK_ENGINE_POPULATION.corrected)
        for formula, share in block.items():
            assert share.key_variable == UK_ENGINE_POPULATION.corrected[formula]
            assert share.factor() == pytest.approx(1.0, rel=1e-12)

    # Clones that drew other keys: each block's c_b / f_b is its share S_b / S
    # of the pool's total, and the shares add up to the whole.
    drawn = [
        (0, _representation_block([10, 11, 12], [0.5, 1.0, 1.5], key_values=[1, 2, 3])),
        (1, _representation_block([12, 10, 11], [1.5, 0.5, 1.0], key_values=[9, 1, 2])),
    ]
    representation = full_measure._engine_population_representation(pool, drawn)
    assert representation["checks"]["weight_share_inputs_match"] is False
    shares = full_measure._block_pool_shares(representation)
    for formula in UK_ENGINE_POPULATION.corrected:
        parts = [
            shares[block][formula].factor() / shares[block][formula].weight_scale
            for block in ("0", "1")
        ]
        assert math.fsum(parts) == pytest.approx(1.0, rel=1e-12)
        assert parts[0] != pytest.approx(parts[1])
        sums = representation["checks"]["weight_share_inputs"][formula]["sum_by_block"]
        assert parts[1] == pytest.approx(sums["1"] / (sums["0"] + sums["1"]))


def test_a_block_without_a_key_column_gets_no_correction(monkeypatch, tmp_path):
    pool = SimpleNamespace(weights_for=lambda entity: SimpleNamespace(total=6.0))
    partial = [
        (0, _representation_block([10, 11, 12], [0.5, 1.0, 1.5])),
        (1, _representation_block([12, 10, 11], [1.5, 0.5, 1.0])),
    ]
    partial[1][1].table("household").drop(columns=["corporate_wealth"], inplace=True)
    representation = full_measure._engine_population_representation(pool, partial)
    assert representation["checks"]["weight_share_inputs"]["corporate_land_value"][
        "missing_columns"
    ] == ["corporate_wealth"]
    assert full_measure._block_pool_shares(representation) is None
    finalized = full_measure._finalize_representation(
        representation,
        [{"mode": "stub"}, {"mode": "stub"}],
        shares_built=False,
    )
    assert finalized["exact"] is False
    assert finalized["share_correction"]["keys_complete"] is False
    assert finalized["share_correction"]["applied_in_every_block"] is False


def _finalize_fixture():
    pool = SimpleNamespace(weights_for=lambda entity: SimpleNamespace(total=6.0))
    blocks = [
        (0, _representation_block([10, 11, 12], [0.5, 1.0, 1.5], key_values=[1, 2, 3])),
        (1, _representation_block([12, 10, 11], [1.5, 0.5, 1.0], key_values=[9, 1, 2])),
    ]
    representation = full_measure._engine_population_representation(pool, blocks)
    shares = full_measure._block_pool_shares(representation)
    receipts = [_clean_block_receipt(shares[block]) for block in ("0", "1")]
    return representation, receipts


def test_clean_corrected_blocks_are_exact_however_their_clones_differ():
    representation, receipts = _finalize_fixture()
    finalized = full_measure._finalize_representation(
        representation, receipts, shares_built=True
    )
    assert finalized["exact"] is True
    assert finalized["exact_basis"] == full_measure.UK_ENGINE_BLOCK_EXACT_BASIS
    # The identical-copies checks fail and stay as diagnostics.
    assert finalized["checks"]["weight_share_inputs_match"] is False
    correction = finalized["share_correction"]
    assert correction["formulas"] == dict(UK_ENGINE_POPULATION.corrected)
    assert correction["periods"] == ["2024", "2025"]
    assert correction["applied_in_every_block"] is True
    assert correction["growth_factors_agree_across_blocks"] is True
    assert correction["max_growth_factor_spread"] == 0.0
    assert correction["corrections_by_block"] == {
        block: receipt["engine_pool_shares"]["corrections"]
        for block, receipt in zip(("0", "1"), receipts, strict=True)
    }
    reach = finalized["population_reach"]
    assert reach["pass"] == "measure"
    assert reach["clean_in_every_block"] is True
    assert reach["declaration_sha256"] == UK_ENGINE_POPULATION.sha256
    assert reach["classes"]["corrected"] == sorted(UK_ENGINE_POPULATION.corrected)
    assert reach["refused"] == []
    assert reach["window"] == {"first_year": 2026, "violations": []}
    assert finalized["annual_income_projection_factors"] == {
        "agree_across_blocks": True,
        "max_rel_spread": 0.0,
    }


def _verdict(receipt, *violations):
    receipt["engine_population_reach"]["clean"] = not violations
    receipt["engine_population_reach"]["violations"] = [
        Violation(*violation).as_receipt() for violation in violations
    ]
    window = [v for v in violations if v[0].startswith("window_")]
    receipt["engine_population_reach"]["window"]["violations"] = [
        Violation(*violation).as_receipt() for violation in window
    ]


def _set(path, value):
    def apply(receipt):
        node = receipt
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value

    return apply


_FINALIZE_FAILURES = [
    pytest.param(change, id=reason)
    for change, reason in (
        (_set(("engine_pool_shares", "applied"), False), "not applied"),
        (_set(("engine_pool_shares", "key_sums_verified"), False), "check (a)"),
        (_set(("engine_pool_shares", "uniform_growth_verified"), False), "check (b)"),
        (_set(("engine_pool_shares", "siblings_match"), False), "siblings moved"),
        (_set(("engine_pool_shares", "siblings_match"), None), "never closed"),
        (_set(("engine_pool_shares", "declaration_valid"), False), "declaration"),
        (_set(("engine_pool_shares", "refusals"), ["refused"]), "a refusal"),
        (_set(("engine_pool_shares", "periods"), ["2024"]), "other periods"),
        (
            _set(("engine_pool_shares", "growth_factors", "2025", "consumption"), 1.04),
            "growth factors disagree",
        ),
        (
            _set(
                (
                    "annual_income_measures",
                    "other_spells_engine_factors",
                    "employment_income_before_lsr",
                ),
                1.05,
            ),
            "annual factors disagree",
        ),
        (
            lambda receipt: _verdict(
                receipt, ("refused_ran", "household_income_decile", "2025")
            ),
            "a refused rank ran",
        ),
        (
            lambda receipt: _verdict(receipt, ("window_period", None, "2026")),
            "the window",
        ),
        (
            lambda receipt: _verdict(
                receipt, ("stored_derived", "attends_private_school", "2025")
            ),
            "a stored input was derived",
        ),
        (lambda receipt: receipt.pop("engine_population_reach"), "no verdict"),
        (lambda receipt: receipt.pop("engine_pool_shares"), "no correction"),
    )
]


@pytest.mark.parametrize("change", _FINALIZE_FAILURES)
def test_exact_follows_every_block_s_correction_and_verdict(change):
    representation, receipts = _finalize_fixture()
    clean = full_measure._finalize_representation(
        representation, copy.deepcopy(receipts), shares_built=True
    )
    assert clean["exact"] is True
    change(receipts[1])
    finalized = full_measure._finalize_representation(
        representation, receipts, shares_built=True
    )
    assert finalized["exact"] is False


def test_a_refused_read_and_the_window_are_named_in_the_representation():
    representation, receipts = _finalize_fixture()
    _verdict(
        receipts[0],
        ("refused_ran", "household_income_decile", "2025"),
        ("window_period", None, "2026"),
    )
    receipts[0]["engine_population_reach"]["refused"] = ["household_income_decile"]
    finalized = full_measure._finalize_representation(
        representation, receipts, shares_built=True
    )
    reach = finalized["population_reach"]
    assert finalized["exact"] is False
    assert reach["clean_in_every_block"] is False
    assert reach["refused"] == ["household_income_decile"]
    assert reach["window"]["violations"] == [
        {"kind": "window_period", "name": None, "period": "2026"}
    ]
    assert list(reach["violations_by_block"]) == ["0"]
    assert {violation["name"] for violation in reach["violations_by_block"]["0"]} == {
        "household_income_decile",
        None,
    }


def test_a_missing_block_receipt_is_never_exact():
    representation, receipts = _finalize_fixture()
    finalized = full_measure._finalize_representation(
        representation, receipts[:1], shares_built=True
    )
    assert finalized["exact"] is False


@pytest.fixture
def fake_engine(monkeypatch):
    """The fake engine's own module, which its declaration names."""

    return install_fake_engine(monkeypatch)


def _shared_block_receipt(block, failure=None):
    """A resolver receipt carrying what the shared EngineBlock adds to it
    (``receipt_entries``, as UKMeasureResolver.receipt does), from one block
    of the fake engine's pool, with one of its checks made to fail."""

    shares, _scales = block_shares(scaled=True)
    own = shares[block]
    simulation = fake_block(block)
    rows = BLOCKS[block]
    if failure == "check_a":
        share = own["share"]
        own = {
            "share": PoolShare(
                "share",
                "key",
                block_key_sum=share.block_key_sum * 1.01,
                pool_key_sum=share.pool_key_sum,
                weight_scale=share.weight_scale,
            )
        }
    elif failure == "check_b":
        simulation.set_input("key_a", P, KEY_A[rows] * GROWTH * [1.0, 1.0, 1.1])
    engine_block = EngineBlock(
        simulation,
        own,
        declaration=fake_declaration(),
        policy=MEASURE_PASS,
        periods=(D, P),
        dataset_year=D,
    )
    sibling = engine_block.wrap(lambda: fake_block(block))()
    if failure == "siblings":
        sibling.set_input("weight", D, np.ones(len(rows)))
    engine_block.close()
    return {
        "mode": "stub",
        "policyengine_uk_version": "test",
        **engine_block.receipt_entries(),
    }


def test_the_shared_blocks_receipts_are_what_the_finalisation_reads(fake_engine):
    """Two blocks the shared EngineBlock corrected and judged clean, on the
    fake engine, make an exact representation: the receipt the resolver
    passes on is the one _finalize_representation reads (microcosm#931)."""

    representation, _receipts = _finalize_fixture()
    receipts = [_shared_block_receipt(block) for block in ("0", "1")]
    finalized = full_measure._finalize_representation(
        representation, receipts, shares_built=True
    )
    assert finalized["exact"] is True
    correction = finalized["share_correction"]
    assert correction["applied_in_every_block"] is True
    assert correction["growth_factors_agree_across_blocks"] is True
    assert correction["refusals_by_block"] == {}
    assert finalized["population_reach"]["clean_in_every_block"] is True


@pytest.mark.parametrize(
    "failure, flag",
    [
        pytest.param("check_a", "key_sums_verified", id="check (a)"),
        pytest.param("check_b", "uniform_growth_verified", id="check (b)"),
        pytest.param("siblings", "siblings_match", id="a sibling moved"),
    ],
)
def test_a_block_whose_check_fails_leaves_the_representation_inexact(
    fake_engine, failure, flag
):
    """A check the shared EngineBlock runs fails in one block: the block
    records it, and the representation is not exact, naming the block and
    holding only names and periods (microcosm#931)."""

    representation, _receipts = _finalize_fixture()
    receipts = [_shared_block_receipt("0"), _shared_block_receipt("1", failure)]
    shares = receipts[1]["engine_pool_shares"]
    assert shares[flag] is False
    finalized = full_measure._finalize_representation(
        representation, receipts, shares_built=True
    )
    assert finalized["exact"] is False
    correction = finalized["share_correction"]
    assert correction[flag] is False
    refusals = correction["refusals_by_block"]
    if failure == "siblings":
        # Recorded by the flag alone: the correction was applied.
        assert shares["applied"] is True and refusals == {}
    else:
        assert shares["applied"] is False
        assert correction["applied_in_every_block"] is False
        assert list(refusals) == ["1"] and len(refusals["1"]) == 1
        assert numbers_are_years(refusals["1"])
