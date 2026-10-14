"""Invented support controls; no survey, Census or administrative files are read."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from microcosm.build import atomic_geography as geo
from microcosm.build.graph_atomic_geography import (
    ATOMIC_GEOGRAPHY_VALIDATION_TYPE,
    atomic_geography_nodes,
    register_atomic_geography_kernels,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactType,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import opaque_artifact_key


def support_parts(system="invented_block", *, district=True):
    columns = {
        name: {
            "kind": "code",
            "source": "invented-lookup-v1",
            "vintage": "invented-2020",
            "relation": "official_tabulation" if name == "district" else "exact",
        }
        for name in ("area", "region", "puma", *(("district",) if district else ()))
    }
    columns.update(
        {
            name: {"kind": "weight", "source": "invented-counts-v1", "basis": basis}
            for name, basis in (("population", "persons"), ("households", "households"))
        }
    )
    arrays = {
        "area": np.asarray(["0001", "0002", "0003", "0004"]),
        "region": np.asarray(["R", "R", "R", "S"]),
        "puma": np.asarray(["A", "A", "B", "C"]),
        "population": np.asarray([0, 2, 6, 1]),
        "households": np.asarray([1, 3, 4, 1]),
    }
    if district:
        arrays["district"] = np.asarray(["01", "02", "02", "03"])
    return {
        "version": 1,
        "system": system,
        "level": "block",
        "code_system": "invented",
        "vintage": "invented-2020",
        "columns": columns,
    }, arrays


def fixture(system="invented_block", *, district=True):
    metadata, arrays = support_parts(system, district=district)
    payload = geo.encode_atomic_support(metadata, arrays)
    support = geo.decode_atomic_support(payload)
    spec = {
        "version": 1,
        "identity": ["source_record"],
        "stream": ["sha256-u53-v1", "invented-geography", 0, 21],
        "outputs": {"area": "block", "system": "area_system", "basis": "area_basis"},
        "systems": [
            {
                "id": system,
                "level": "block",
                "code_system": "invented",
                "vintage": "invented-2020",
                "source": system + "_source",
                "selector": {},
                "constraints": [
                    {"input": "observed_region", "support": "region", "required": True},
                    {"input": "observed_puma", "support": "puma", "required": False},
                ],
                "observed_area": None,
                "stages": [{"level": "area", "weight": "population"}],
                "layers": (
                    [
                        {
                            "input": "district",
                            "output": "district_code",
                            "vintage": "invented-2020",
                            "relation": "official_tabulation",
                            "source": "invented-lookup-v1",
                        }
                    ]
                    if district
                    else []
                ),
            }
        ],
    }
    households = pd.DataFrame(
        {
            "household_id": np.arange(1, 9, dtype=np.int64),
            "source_record": pd.array(
                ["source-" + str(i) for i in range(8)], dtype="string"
            ),
            "observed_region": pd.array(["R"] * 7 + ["S"], dtype="string"),
            "observed_puma": pd.array(
                [None, "A", "B", None, "A", None, "B", "C"], dtype="string"
            ),
        }
    )
    return households, spec, {system: support}, payload


def complete(households, spec, supports):
    assigned = pd.concat(
        [households, geo.assign_atomic(households, spec, supports)], axis=1
    )
    return pd.concat([assigned, geo.derive_geography(assigned, spec, supports)], axis=1)


def test_codec_is_deterministic_and_decoded_values_are_immutable():
    metadata, arrays = support_parts()
    first = geo.encode_atomic_support(metadata, arrays)
    assert first == geo.encode_atomic_support(
        metadata, dict(reversed(list(arrays.items())))
    )
    support = geo.decode_atomic_support(first)
    assert support.arrays["area"].tolist() == ["0001", "0002", "0003", "0004"]
    with pytest.raises(ValueError):
        support.arrays["area"].setflags(write=True)
    with pytest.raises(TypeError):
        support.metadata["columns"]["area"]["source"] = "changed"


def test_integer_inverse_cdf_respects_zero_mass_and_boundaries():
    _, _, supports, _ = fixture()
    index = geo._SupportIndex(supports["invented_block"])
    stage = {"level": "area", "weight": "population"}
    cell = (("region", "R"),)
    assert index.pick(cell, stage, 0.0) == "0002"
    assert index.pick(cell, stage, 0.25 - 1 / 2**53) == "0002"
    assert index.pick(cell, stage, 0.25) == "0003"
    assert index.pick(cell, stage, 1 - 1 / 2**53) == "0003"


def test_draws_are_stable_under_order_subset_extra_columns_and_unrelated_rows():
    households, spec, supports, _ = fixture()
    before = households.copy(deep=True)
    expected = geo.assign_atomic(households, spec, supports)
    reordered = households.iloc[[7, 2, 0, 5]]
    actual = geo.assign_atomic(reordered.assign(unrelated=42), spec, supports)
    pd.testing.assert_frame_equal(actual, expected.loc[reordered.index])
    extra = households.iloc[[0]].copy()
    extra.index = [100]
    extra["source_record"] = "unrelated-record"
    grown = geo.assign_atomic(pd.concat([households, extra]), spec, supports)
    pd.testing.assert_frame_equal(grown.loc[households.index], expected)
    pd.testing.assert_frame_equal(households, before)


def test_stage_law_uses_declared_weights_at_each_level():
    households, spec, supports, _ = fixture()
    spec["systems"][0]["stages"] = [
        {"level": "puma", "weight": "households"},
        {"level": "area", "weight": "population"},
    ]
    result = complete(households, spec, supports)
    assert result.loc[1, "block"] == "0002"
    assert result.loc[2, "block"] == "0003"
    assert geo.validate_geography(result, spec, supports)["outcome"] == "pass"
    # A/B household mass is 4/4, but population mass is 2/6.
    index = geo._SupportIndex(supports["invented_block"])
    assert index.pick((("region", "R"),), spec["systems"][0]["stages"][0], 0.49) == "A"
    assert index.pick((("region", "R"),), spec["systems"][0]["stages"][0], 0.5) == "B"


def test_qualified_observed_area_is_retained_even_with_zero_sampling_mass():
    households, spec, supports, _ = fixture()
    households["observed_block"] = pd.array(["0001"] + [None] * 7, dtype="string")
    spec["systems"][0]["observed_area"] = "observed_block"
    result = complete(households, spec, supports)
    assert result.loc[0, "block"] == "0001"
    assert result.loc[0, "area_basis"] == "observed"
    assert result.loc[0, "district_code"] == "01"
    assert geo.validate_geography(result, spec, supports)["outcome"] == "pass"


@pytest.mark.parametrize(
    "defect",
    [
        "missing_region",
        "wrong_region",
        "inconsistent_puma",
        "duplicate_identity",
        "null_identity",
        "existing_output",
        "unknown_observed_area",
        "conflicting_observed_area",
    ],
)
def test_assignment_refuses_invalid_or_conflicting_households(defect):
    households, spec, supports, _ = fixture()
    if defect == "missing_region":
        households.loc[0, "observed_region"] = pd.NA
    elif defect == "wrong_region":
        households.loc[0, "observed_region"] = "missing"
    elif defect == "inconsistent_puma":
        households.loc[0, "observed_puma"] = "C"
    elif defect == "duplicate_identity":
        households.loc[0, "source_record"] = households.loc[1, "source_record"]
    elif defect == "null_identity":
        households.loc[0, "source_record"] = pd.NA
    elif defect == "existing_output":
        households["block"] = "untouched"
    else:
        households["observed_block"] = pd.array(
            ["unknown" if defect == "unknown_observed_area" else "0004"] + [None] * 7,
            dtype="string",
        )
        spec["systems"][0]["observed_area"] = "observed_block"
    with pytest.raises(ValueError):
        geo.assign_atomic(households, spec, supports)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_area",
        "negative_weight",
        "float_weight",
        "bool_weight",
        "overflow_weight",
        "all_zero_weight",
        "unknown_relation",
        "wrong_area_vintage",
        "empty_code",
        "wrong_length",
    ],
)
def test_support_refuses_ambiguous_or_invalid_artifacts(defect):
    metadata, arrays = support_parts()
    if defect == "duplicate_area":
        arrays["area"][1] = arrays["area"][0]
    elif defect == "negative_weight":
        arrays["population"][0] = -1
    elif defect == "float_weight":
        arrays["population"] = arrays["population"].astype(float)
    elif defect == "bool_weight":
        arrays["population"] = arrays["population"].astype(bool)
    elif defect == "overflow_weight":
        arrays["population"] = np.asarray([2**63] * 4, dtype=np.uint64)
    elif defect == "all_zero_weight":
        arrays["population"] *= 0
    elif defect == "unknown_relation":
        metadata["columns"]["district"]["relation"] = "exact_enough"
    elif defect == "wrong_area_vintage":
        metadata["columns"]["area"]["vintage"] = "different"
    elif defect == "empty_code":
        arrays["area"][0] = ""
    elif defect == "wrong_length":
        arrays["region"] = arrays["region"][:-1]
    with pytest.raises(ValueError):
        geo.encode_atomic_support(metadata, arrays)


@pytest.mark.parametrize("field", ["source", "relation", "vintage"])
def test_layer_metadata_must_match_the_pinned_support(field):
    households, spec, supports, _ = fixture()
    spec["systems"][0]["layers"][0][field] = (
        "exact" if field == "relation" else "different"
    )
    with pytest.raises(ValueError):
        geo.assign_atomic(households, spec, supports)


def test_three_nation_systems_use_the_same_operations_and_mark_absent_layers():
    households, spec, supports, _ = fixture()
    households["nation"] = pd.array(
        ["one"] * 3 + ["two"] * 3 + ["three"] * 2, dtype="string"
    )
    spec["systems"][0]["selector"] = {"nation": ["one"]}
    for name, selector, district in (
        ("invented_oa", "two", True),
        ("invented_dz", "three", False),
    ):
        _, other, added, _ = fixture(name, district=district)
        system = other["systems"][0]
        system["selector"] = {"nation": [selector]}
        spec["systems"].append(system)
        supports.update(added)
    result = complete(households, spec, supports)
    assert (
        result["area_system"].tolist()
        == ["invented_block"] * 3 + ["invented_oa"] * 3 + ["invented_dz"] * 2
    )
    assert result.loc[[6, 7], "district_code"].isna().all()
    assert geo.validate_geography(result, spec, supports)["outcome"] == "pass"
    spec["systems"][1]["selector"] = {"nation": ["one", "two"]}
    with pytest.raises(ValueError, match="exactly one"):
        geo.assign_atomic(households, spec, supports)


def test_clone_inheritance_and_pruned_mapping_checks_do_not_redraw():
    households, spec, supports, _ = fixture()
    original = complete(households, spec, supports)
    clone = original.iloc[[2, 0, 5]].copy()
    clone["household_id"] += 100
    clone["source_record"] += "-clone"
    assert geo.validate_geography(clone, spec, supports)["households"] == 3
    clone.iloc[0, clone.columns.get_loc("district_code")] = "wrong"
    with pytest.raises(ValueError, match="derived geography differs"):
        geo.validate_geography(clone, spec, supports)


@pytest.mark.parametrize("defect", ["atomic", "basis", "observed", "system"])
def test_final_integrity_gate_refuses_changed_geography(defect):
    households, spec, supports, _ = fixture()
    result = complete(households, spec, supports)
    if defect == "atomic":
        result.loc[0, "block"] = "missing"
    elif defect == "basis":
        result.loc[0, "area_basis"] = "observed"
    elif defect == "observed":
        result.loc[0, "observed_region"] = "wrong"
    elif defect == "system":
        result.loc[0, "area_system"] = pd.NA
    with pytest.raises(ValueError):
        geo.validate_geography(result, spec, supports)


class _InventedSpine(KernelBase):
    ref = "invented.atomic_spine@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        households, _, _, _ = fixture()
        person = pd.DataFrame(
            {
                "person_id": np.arange(1, 9, dtype=np.int64),
                "person_household_id": households["household_id"],
                "age": np.arange(21, 29, dtype=np.int64),
            }
        )
        return KernelResult(
            frame=Frame(
                {"person": person, "household": households},
                EntitySchema(group_entities=("household",)),
                {"household": Weights(np.ones(8), WeightKind.DESIGN)},
                pd.Series(["invented"] * 8, name="stratum"),
            )
        )


@pytest.mark.parametrize("emit_validation_artifact", (False, True))
def test_real_executor_cold_warm_and_changed_source_identity(
    tmp_path, emit_validation_artifact
):
    households, spec, supports, payload = fixture()
    columns = (
        Owned("person", "age", "int64"),
        *(
            Owned("household", name, "string")
            for name in ("source_record", "observed_region", "observed_puma")
        ),
    )
    base = Node(
        "spine",
        _InventedSpine.ref,
        sources=("invented_spine",),
        structural=StructuralDelta.CREATE,
        outputs=columns,
    )
    nodes = atomic_geography_nodes(
        spec, columns, base=base.id, emit_validation_artifact=emit_validation_artifact
    )
    graph = compile_graph(
        Graph(
            "invented",
            (
                SourceRef("invented_spine", "raw-bytes-v1"),
                SourceRef("invented_block_source", "raw-bytes-v1"),
            ),
            (base, *nodes),
        )
    )
    registry = KernelRegistry()
    registry.register(_InventedSpine())
    register_atomic_geography_kernels(registry)
    raw_spine = tmp_path / "invented-spine.bin"
    raw_spine.write_bytes(b"invented test source declaration")
    raw_support = tmp_path / "invented-support.npz"
    raw_support.write_bytes(payload)
    sources = {"invented_spine": raw_spine, "invented_block_source": raw_support}
    store = ContentStore(tmp_path / "store")
    cold = run_graph(graph, sources=sources, store=store, kernels=registry)
    warm = run_graph(
        graph, sources=sources, store=store, kernels=registry, resume="require"
    )
    actual = cold.population(base.id).table("household")
    expected = complete(households, spec, supports)
    for column in expected:
        left, right = actual[column], expected[column]
        if isinstance(expected[column].dtype, pd.StringDtype):
            # Source checkpoints and emitted columns may use different string
            # backends. Check the nullable-string contract, then compare every
            # value and missing cell using one explicit physical representation.
            assert isinstance(left.dtype, pd.StringDtype)
            assert left.dtype.na_value is pd.NA
            left = left.astype(pd.StringDtype(storage="python"))
            right = right.astype(pd.StringDtype(storage="python"))
        pd.testing.assert_series_equal(left, right, check_index_type=False)
    assert cold.nodes["geography.assign"].key == warm.nodes["geography.assign"].key
    assert cold.nodes["geography.gate"].receipt["outcome"] == "pass"
    gate = cold.nodes["geography.gate"]
    if emit_validation_artifact:
        output = graph.graph.node("geography.gate").artifact_outputs
        assert len(output) == 1 and output[0].type == ATOMIC_GEOGRAPHY_VALIDATION_TYPE
        key = gate.opaque_artifacts["validation"]
        assert key == opaque_artifact_key(gate.key, "validation")
        assert store.load_bytes(key) == canonical_json(
            geo.validate_geography(expected, spec, supports)
        )
        assert warm.nodes["geography.gate"].opaque_artifacts["validation"] == key
    else:
        assert not gate.opaque_artifacts

    np.testing.assert_array_equal(
        cold.population(base.id).weights_for("household").values, np.ones(8)
    )
    metadata, arrays = support_parts()
    arrays["population"][1] += 1
    raw_support.write_bytes(geo.encode_atomic_support(metadata, arrays))
    changed = run_graph(graph, sources=sources, store=store, kernels=registry)
    assert changed.nodes["geography.assign"].key != cold.nodes["geography.assign"].key


def test_graph_builder_refuses_input_overwrites_and_duplicate_inventory():
    _, spec, _, _ = fixture()
    columns = tuple(
        Owned("household", c, "string")
        for c in ("source_record", "observed_region", "observed_puma")
    )
    with pytest.raises(ValueError, match="overwrite"):
        atomic_geography_nodes(
            spec, (*columns, Owned("household", "block", "string")), base="spine"
        )
    with pytest.raises(ValueError, match="repeats"):
        atomic_geography_nodes(spec, (*columns, columns[0]), base="spine")
    with pytest.raises(ValueError, match="missing"):
        atomic_geography_nodes(spec, columns[1:], base="spine")
    altered = deepcopy(spec)
    altered["outputs"]["area"] = "observed_region"
    with pytest.raises(ValueError, match="overwritten"):
        atomic_geography_nodes(altered, columns, base="spine")


@pytest.mark.parametrize("defect", ("missing", "type"))
def test_typed_geography_gate_edge_refuses_invalid_producer_contract(defect):
    _, spec, _, _ = fixture()
    columns = tuple(
        Owned("household", name, "string")
        for name in ("source_record", "observed_region", "observed_puma")
    )
    base = Node(
        "spine",
        _InventedSpine.ref,
        sources=("invented_spine",),
        structural=StructuralDelta.CREATE,
        outputs=columns,
    )
    nodes = atomic_geography_nodes(
        spec, columns, base=base.id, emit_validation_artifact=(defect != "missing")
    )
    edge_type = (
        ArtifactType("invented.wrong_gate", 1)
        if defect == "type"
        else ATOMIC_GEOGRAPHY_VALIDATION_TYPE
    )
    consumer = Node(
        "consumer",
        "invented.consumer@1",
        population=base.id,
        artifact_inputs=(
            ArtifactInput("validation", "geography.gate", "validation", edge_type),
        ),
    )
    with pytest.raises(ValueError, match="no declared artifact|type does not match"):
        compile_graph(
            Graph(
                "invented",
                (
                    SourceRef("invented_spine", "raw-bytes-v1"),
                    SourceRef("invented_block_source", "raw-bytes-v1"),
                ),
                (base, *nodes, consumer),
            )
        )
