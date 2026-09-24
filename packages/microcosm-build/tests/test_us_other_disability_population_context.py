"""Literal Population/manifest comparison seams; no owners, QRF or graph run."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_other_disability_filter_reconstruction import _incoming

from microcosm.build.us_runtime import graph_us_other_disability_host as host
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph import ContentStore, KernelResult, Node, StructuralDelta
from microcosm.graph.decl import Owned, WeightTransition


def _receiving():
    source = _incoming()
    transition = Node(
        "invented.importance",
        "test@1",
        base=source.version,
        structural=StructuralDelta.REWEIGHT,
        mass="conserve",
        weights=WeightTransition("household", "importance", mass="conserve"),
    )
    population = host.population_ops.patch(
        source,
        transition,
        KernelResult(
            weights=Weights(
                source.frame.weights_for("household").values, WeightKind.IMPORTANCE
            )
        ),
    )
    writer = Node(
        "invented.writer",
        "test@1",
        population=population.version,
        outputs=(Owned("person", "prediction", "float64"),),
    )
    return host.population_ops.patch(
        population,
        writer,
        KernelResult(
            columns={
                ("person", "prediction"): pd.Series(
                    [0.0, 10.0, 20.0], index=[10, 20, 30], dtype="float64"
                )
            }
        ),
    )


def _manifest(population, *, frame=None, ledger=None):
    """Unissued literal view: only the real manifest's Frame/ledger surface."""
    frames = {population.version: population.frame if frame is None else frame}
    ledgers = {population.version: population.mass_ledger if ledger is None else ledger}
    return SimpleNamespace(
        populations=frames,
        population=frames.__getitem__,
        mass_ledger=ledgers.__getitem__,
    )


def _predecessor(population, manifest):
    # The helper grants no run/source authority. Its production caller retains
    # and checks the genuine predecessor before reaching this comparison.
    return SimpleNamespace(
        population=population,
        manifest=manifest,
        compiled=SimpleNamespace(
            order=("invented.writer",),
            versions={"invented.writer": population.version},
        ),
    )


def test_manifest_comparison_preserves_valid_nontrivial_execution_context():
    population = _receiving()
    assert population.owners["person", "prediction"] == "invented.writer"
    assert population.version == "invented.importance"
    assert population.weight_kind["household"] is WeightKind.IMPORTANCE
    assert tuple(population.design_weights) == ("household",)
    assert len(population.mass_ledger) == 1
    manifest = _manifest(population)
    # Reproduce the actual failure: a manifest Frame cannot recreate ownership
    # and retained DESIGN anchors through from_frame defaults.
    with pytest.raises(ValueError, match="SURVEY_POPULATION_REPLAY_POPULATION_CONTEXT"):
        host.physical.replay.same_replayed_population(
            population,
            host.population_ops.Population.from_frame(
                manifest.population(population.version),
                population.version,
                mass_ledger=manifest.mass_ledger(population.version),
            ),
        )
    host._compare_manifest_population(population, manifest, population.version)
    host._compare_predecessor_populations(
        _predecessor(population, manifest),
        {"invented.writer": population},
        "invented.writer",
    )


@pytest.mark.parametrize("defect", ["owner", "design", "ledger", "version"])
def test_full_retained_receiving_context_is_not_normalized_away(defect):
    population = _receiving()
    if defect == "owner":
        owners = dict(population.owners)
        owners["person", "prediction"] = "invented.intruder"
        changed = replace(population, owners=owners)
        reason = "POPULATION_CONTEXT"
    elif defect == "design":
        changed = replace(
            population, design_weights={"household": np.array([1.6, 4.0])}
        )
        reason = "DESIGN_BYTES"
    elif defect == "ledger":
        changed = replace(
            population,
            mass_ledger=(
                replace(population.mass_ledger[0], node_id="invented.intruder"),
            ),
        )
        reason = "POPULATION_CONTEXT"
    else:
        changed = replace(population, version="invented.intruder")
        reason = "POPULATION_CONTEXT"
    with pytest.raises(ValueError, match="SURVEY_POPULATION_REPLAY_" + reason):
        host._compare_predecessor_populations(
            _predecessor(population, _manifest(population)),
            {"invented.writer": changed},
            "invented.writer",
        )


@pytest.mark.parametrize("defect", ["value", "weight", "ledger"])
def test_manifest_frame_and_ledger_mutations_still_refuse(defect):
    population = _receiving()
    tables = {
        entity: population.frame.table(entity).copy()
        for entity in population.frame.entities
    }
    weights = {"household": population.frame.weights_for("household")}
    ledger = population.mass_ledger
    if defect == "value":
        tables["person"].loc[8, "prediction"] = 11.0
        reason = "NATIVE_BITS"
    elif defect == "weight":
        weights["household"] = Weights(np.array([1.6, 4.0]), WeightKind.IMPORTANCE)
        reason = "WEIGHT_BYTES"
    else:
        ledger = (replace(ledger[0], after_total=8.0),)
        reason = "POPULATION_CONTEXT"
    frame = Frame(
        tables,
        population.frame.schema,
        weights,
        population.frame.strata.copy(),
        mass_log=population.frame.mass_log,
    )
    with pytest.raises(ValueError, match="SURVEY_POPULATION_REPLAY_" + reason):
        host._compare_manifest_population(
            population,
            _manifest(population, frame=frame, ledger=ledger),
            population.version,
        )


def _nullable_roundtrip(tmp_path, dtype):
    """Real storage normalization on literal populations, without source authority."""
    population = _receiving()
    mask = np.array([False, True, True])
    values = (
        pd.arrays.IntegerArray(np.array([11, 71, 73], dtype=np.int64), mask)
        if dtype == "Int64"
        else pd.arrays.BooleanArray(np.array([True, True, True]), mask)
    )
    tables = {
        entity: population.frame.table(entity).copy(deep=True)
        for entity in population.frame.entities
    }
    tables["person"]["nullable_observation"] = values
    frame = Frame(
        tables,
        population.frame.schema,
        {
            entity: population.frame.weights_for(entity)
            for entity in population.frame.weighted_entities
        },
        population.frame.strata.copy(),
        mass_log=population.frame.mass_log,
        metadata=population.frame.metadata,
    )
    original = replace(
        population,
        frame=frame,
        owners={
            **population.owners,
            ("person", "nullable_observation"): "invented.nullable_writer",
        },
    )
    before = frame.person.nullable_observation.array._data.tobytes()
    store = ContentStore(tmp_path / "invented-nullable-store")
    store.put_frame("a" * 64, frame)
    replayed = replace(original, frame=store.load_frame("a" * 64))
    assert frame.person.nullable_observation.array._data.tobytes() == before
    cached = replayed.frame.person.nullable_observation.array
    assert bool(np.all(cached._data[cached._mask] == 0))
    assert cached._data.tobytes() != before
    return original, replayed


@pytest.mark.parametrize("dtype", ["Int64", "boolean"])
@pytest.mark.parametrize("boundary", ["retained", "reconstructed"])
def test_manifest_comparison_accepts_directional_nullable_store_roundtrip(
    tmp_path, dtype, boundary
):
    original, replayed = _nullable_roundtrip(tmp_path, dtype)
    if boundary == "retained":
        manifest = _manifest(original)
        host.host._compare_inherited_manifest_population(
            expected_manifest=manifest,
            replayed_population=replayed,
            version=original.version,
        )
        host._compare_predecessor_populations(
            _predecessor(original, manifest),
            {"invented.writer": replayed},
            "invented.writer",
        )
    else:
        host._compare_manifest_population(
            original, _manifest(replayed), original.version
        )


@pytest.mark.parametrize("dtype", ["Int64", "boolean"])
@pytest.mark.parametrize("defect", ["present", "mask", "null_backing"])
@pytest.mark.parametrize("boundary", ["retained", "reconstructed"])
def test_manifest_comparison_refuses_nullable_replay_mutations(
    tmp_path, dtype, defect, boundary
):
    original, replayed = _nullable_roundtrip(tmp_path, dtype)
    changed = replayed.frame.person.nullable_observation.array
    if defect == "present":
        changed._data[0] = 12 if dtype == "Int64" else False
        reason = "PRESENT_BITS"
    elif defect == "mask":
        changed._mask[0] = True
        reason = "MASKED_STORAGE"
    else:
        # Neither the original backing nor the store's canonical zero backing.
        changed._data[1] = 99 if dtype == "Int64" else True
        reason = "NONCANONICAL_NULL_BACKING"
    with pytest.raises(ValueError, match="SURVEY_POPULATION_REPLAY_" + reason):
        if boundary == "retained":
            host.host._compare_inherited_manifest_population(
                expected_manifest=_manifest(original),
                replayed_population=replayed,
                version=original.version,
            )
        else:
            host._compare_manifest_population(
                original, _manifest(replayed), original.version
            )
