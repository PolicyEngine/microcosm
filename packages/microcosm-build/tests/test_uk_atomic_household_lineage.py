"""Supplied lineage and real invented Graph/Frame controls; no FRS or engine."""

import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest
from test_uk_atomic_area_support import payloads

from microcosm.build import atomic_geography
from microcosm.build.graph_atomic_geography import (
    atomic_geography_nodes,
    register_atomic_geography_kernels,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    IDENTITY_COLUMN,
    SOURCES,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.atomic_household_lineage import (
    HouseholdExpansion,
    HouseholdSelection,
    project_atomic_household_keys,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)


def roots(ids=(1, 2)):
    return pd.DataFrame(
        {
            "household_id": np.asarray(ids, dtype=np.int64),
            "source": pd.array(["invented-frs"] * len(ids), dtype="string"),
            "source_vintage": pd.array(["invented-2024-25"] * len(ids), dtype="string"),
            "source_household_id": np.asarray(ids, dtype=np.int64),
        }
    )


def expansion(branch, before, pairs, ordinals=None):
    return HouseholdExpansion(
        branch=branch,
        before_ids=tuple(before),
        after_ids=(*before, *(c for c, _ in pairs)),
        parent_pairs=tuple(pairs),
        child_ordinals=tuple((c, 1) for c, _ in pairs)
        if ordinals is None
        else ordinals,
    )


def chain():
    a = expansion("spi_support_channel", (1, 2), ((11, 1), (12, 2)))
    b = expansion("cgt_incidence_clone", a.after_ids, ((21, 1), (31, 11)))
    c = expansion("cgt_band_donors", b.after_ids, ((41, 31),))
    s = HouseholdSelection(c.after_ids, (1, 11, 21, 31, 41))
    d = expansion("geographic_support", s.after_ids, ((101, 1), (141, 41)))
    return a, b, c, s, d


def project(source=None, steps=None, ids=None):
    source = roots() if source is None else source
    steps = chain() if steps is None else steps
    ids = steps[-1].after_ids if ids is None else ids
    return project_atomic_household_keys(
        source, steps=steps, final_ids=np.asarray(ids, dtype=np.int64)
    )


def by_id(table):
    return table.set_index("household_id")[IDENTITY_COLUMN].sort_index()


def test_explicit_full_chain_selection_and_clone_keys():
    result = project()
    assert tuple(result.household_id) == chain()[-1].after_ids
    assert list(result) == ["household_id", IDENTITY_COLUMN]
    assert result.household_id.dtype == np.dtype("int64")
    assert result[IDENTITY_COLUMN].dtype == pd.StringDtype(storage="python")
    expected_paths = {
        1: (),
        11: (("spi_support_channel", 1),),
        21: (("cgt_incidence_clone", 1),),
        31: (("spi_support_channel", 1), ("cgt_incidence_clone", 1)),
        41: (
            ("spi_support_channel", 1),
            ("cgt_incidence_clone", 1),
            ("cgt_band_donors", 1),
        ),
        101: (("geographic_support", 1),),
        141: (
            ("spi_support_channel", 1),
            ("cgt_incidence_clone", 1),
            ("cgt_band_donors", 1),
            ("geographic_support", 1),
        ),
    }
    actual = by_id(result)
    for target, path in expected_paths.items():
        assert actual[target] == household_draw_key(
            source="invented-frs",
            source_vintage="invented-2024-25",
            source_household_id=1,
            clone_path=path,
        )
    assert actual.is_unique


def test_axes_and_pair_order_do_not_supply_identity():
    steps = tuple(
        replace(
            x,
            before_ids=x.before_ids[::-1],
            after_ids=x.after_ids[::-1],
            **(
                {
                    "parent_pairs": x.parent_pairs[::-1],
                    "child_ordinals": x.child_ordinals[::-1],
                }
                if isinstance(x, HouseholdExpansion)
                else {}
            ),
        )
        for x in chain()
    )
    result = project(roots().iloc[::-1], steps, steps[-1].after_ids)
    pd.testing.assert_series_equal(by_id(result), by_id(project()))


def test_exact_large_ids_and_growth_preserve_existing_keys():
    big = 2**53 + 7
    first = expansion("geographic_support", (big,), ((big + 2, big),))
    more = expansion(
        "geographic_support",
        (big, big + 1),
        ((big + 2, big), (big + 3, big + 1)),
        ((big + 2, 1), (big + 3, 9)),
    )
    small = project(roots((big,)), (first,))
    large = project(roots((big, big + 1)), (more,))
    pd.testing.assert_series_equal(by_id(small), by_id(large).loc[by_id(small).index])
    assert by_id(large).is_unique
    assert str(big) in by_id(small).iloc[0]


def test_source_and_explicit_ordinal_change_keys_without_numeric_inference():
    expected = by_id(project())
    source = roots()
    source.loc[0, "source_vintage"] = "different-vintage"
    changed = by_id(project(source))
    assert (changed != expected).all()
    steps = chain()
    changed = by_id(
        project(
            steps=(*steps[:-1], replace(steps[-1], child_ordinals=((101, 2), (141, 8))))
        )
    )
    assert (changed.loc[[101, 141]] != expected.loc[[101, 141]]).all()
    pd.testing.assert_series_equal(changed.drop([101, 141]), expected.drop([101, 141]))


def test_noop_and_empty_selection_are_explicit_and_detached():
    original = roots()
    saved = original.copy(deep=True)
    no_steps = project(original, (), (2, 1))
    assert no_steps.household_id.tolist() == [2, 1]
    empty = project(original, (HouseholdSelection((1, 2), ()),), ())
    assert empty.empty and empty.household_id.dtype == np.dtype("int64")
    no_steps.loc[0, IDENTITY_COLUMN] = "changed descriptive output"
    pd.testing.assert_frame_equal(original, saved)
    with pytest.raises(FrozenInstanceError):
        chain()[0].branch = "changed"


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_id",
        "float_id",
        "bool_id",
        "zero_id",
        "duplicate_root_identity",
        "missing_source",
        "blank_source",
        "extra_column",
        "duplicate_column",
        "invalid_original",
        "empty_roots",
    ],
)
def test_roots_refuse_unqualified_or_ambiguous_descriptions(defect):
    source = roots()
    if defect == "duplicate_id":
        source.loc[1, "household_id"] = 1
    elif defect == "float_id":
        source["household_id"] = source.household_id.astype(float)
    elif defect == "bool_id":
        source["household_id"] = [True, False]
    elif defect == "zero_id":
        source.loc[0, "household_id"] = 0
    elif defect == "duplicate_root_identity":
        source.loc[1, "source_household_id"] = 1
    elif defect == "missing_source":
        source.loc[0, "source"] = pd.NA
    elif defect == "blank_source":
        source.loc[0, "source_vintage"] = " "
    elif defect == "extra_column":
        source["guess"] = 1
    elif defect == "duplicate_column":
        source.columns = ["household_id", "source", "source", "source_household_id"]
    elif defect == "invalid_original":
        source["source_household_id"] = [1.0, 2.0]
    elif defect == "empty_roots":
        source = source.iloc[:0]
    with pytest.raises((TypeError, ValueError)):
        project(source)


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_branch",
        "reordered_branch",
        "repeated_branch",
        "mutable_steps",
        "mutable_axis",
        "duplicate_before",
        "duplicate_after",
        "unexplained_before",
        "deleted_incumbent",
        "unknown_parent",
        "null_parent",
        "same_step_parent",
        "missing_parent",
        "duplicate_parent_target",
        "unexplained_child",
        "missing_ordinal",
        "extra_ordinal",
        "duplicate_ordinal",
        "negative_ordinal",
        "bool_ordinal",
        "float_child",
        "overflow_child",
        "selection_arrival",
        "implicit_selection",
        "final_extra",
        "final_duplicate",
    ],
)
def test_invalid_steps_and_final_axis_refuse(defect):
    steps = list(chain())
    first = steps[0]
    final = steps[-1].after_ids
    if defect == "unknown_branch":
        steps[0] = replace(first, branch="income_guess")
    elif defect == "reordered_branch":
        steps[0] = replace(first, branch="geographic_support")
    elif defect == "repeated_branch":
        steps[1] = replace(steps[1], branch=first.branch)
    elif defect == "mutable_steps":
        with pytest.raises((TypeError, ValueError)):
            project(steps=steps)
        return
    elif defect == "mutable_axis":
        steps[0] = replace(first, before_ids=[1, 2])
    elif defect == "duplicate_before":
        steps[0] = replace(first, before_ids=(1, 1, 2))
    elif defect == "duplicate_after":
        steps[0] = replace(first, after_ids=(*first.after_ids, 11))
    elif defect == "unexplained_before":
        steps[0] = replace(first, before_ids=(1, 3))
    elif defect == "deleted_incumbent":
        steps[0] = replace(first, after_ids=(1, 11, 12))
    elif defect == "unknown_parent":
        steps[0] = replace(first, parent_pairs=((11, 3), (12, 2)))
    elif defect == "null_parent":
        steps[0] = replace(first, parent_pairs=((11, None), (12, 2)))
    elif defect == "same_step_parent":
        steps[0] = replace(first, parent_pairs=((11, 1), (12, 11)))
    elif defect == "missing_parent":
        steps[0] = replace(first, parent_pairs=((11, 1),))
    elif defect == "duplicate_parent_target":
        steps[0] = replace(first, parent_pairs=((11, 1), (11, 2)))
    elif defect == "unexplained_child":
        steps[0] = replace(first, after_ids=(*first.after_ids, 13))
    elif defect == "missing_ordinal":
        steps[0] = replace(first, child_ordinals=((11, 1),))
    elif defect == "extra_ordinal":
        steps[0] = replace(first, child_ordinals=(*first.child_ordinals, (99, 1)))
    elif defect == "duplicate_ordinal":
        steps[0] = replace(first, parent_pairs=((11, 1), (12, 1)))
    elif defect == "negative_ordinal":
        steps[0] = replace(first, child_ordinals=((11, -1), (12, 1)))
    elif defect == "bool_ordinal":
        steps[0] = replace(first, child_ordinals=((11, True), (12, 1)))
    elif defect == "float_child":
        steps[0] = replace(first, parent_pairs=((11.0, 1), (12, 2)))
    elif defect == "overflow_child":
        steps[0] = replace(
            first,
            after_ids=(1, 2, 11, 2**63),
            parent_pairs=((11, 1), (2**63, 2)),
            child_ordinals=((11, 1), (2**63, 1)),
        )
    elif defect == "selection_arrival":
        steps[3] = replace(steps[3], after_ids=(*steps[3].after_ids, 99))
    elif defect == "implicit_selection":
        del steps[3]
    elif defect == "final_extra":
        final = (*final, 99)
    elif defect == "final_duplicate":
        final = (*final, final[0])
    with pytest.raises((TypeError, ValueError)):
        project(steps=tuple(steps), ids=final)


class _FrameKernel(KernelBase):
    ref = "invented.uk.lineage.frame@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def __init__(self, frame):
        self.frame = frame

    def run(self, context):
        return KernelResult(frame=self.frame)


class _ExpandKernel(KernelBase):
    ref = "invented.uk.lineage.expand@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.EXPAND
    )

    def run(self, context):
        return KernelResult(
            expand={
                e: pd.Series(
                    [1, 2, 3],
                    index=pd.Index([11, 12, 13], name=e + "_id"),
                    dtype="int64",
                )
                for e in ("person", "benunit", "household")
            },
            weights=Weights(np.full(6, 0.5), WeightKind.IMPORTANCE),
        )


class _SelectKernel(KernelBase):
    ref = "invented.uk.lineage.select@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER
    )

    def run(self, context):
        ids = context.tables["person"].person_id
        return KernelResult(
            keep=pd.Series(
                ids.isin([1, 3, 11, 13]).to_numpy(),
                index=pd.Index(ids, name="person_id"),
            )
        )


def test_actual_expand_filter_receipts_feed_shared_atomic_graph_and_replay(tmp_path):
    ids = np.arange(1, 4, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_benunit_id": ids,
            "person_household_id": ids,
            "age": [21, 33, 44],
        }
    )
    benunit = pd.DataFrame({"benunit_id": ids, "family_marker": [4, 5, 6]})
    household = pd.DataFrame(
        {
            "household_id": ids,
            "region": pd.array(
                ["LONDON", "SCOTLAND", "NORTHERN_IRELAND"], dtype="string"
            ),
        }
    )
    frame = Frame(
        {"person": person, "benunit": benunit, "household": household},
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.ones(3), WeightKind.DESIGN)},
        pd.Series(["invented"] * 3, name="stratum"),
        metadata={"note": "invented UK lineage"},
    )
    columns = (
        Owned("person", "age", "int64"),
        Owned("benunit", "family_marker", "int64"),
        Owned("household", "region", "string"),
    )
    graph = compile_graph(
        Graph(
            "uk",
            (SourceRef("invented", "raw-bytes-v1"),),
            (
                Node(
                    "root",
                    _FrameKernel.ref,
                    structural=StructuralDelta.CREATE,
                    sources=("invented",),
                    outputs=columns,
                ),
                Node(
                    "expand",
                    _ExpandKernel.ref,
                    base="root",
                    structural=StructuralDelta.EXPAND,
                    mass="conserve",
                    params={
                        "expand_cells": (),
                        "expand_weight_entity": "household",
                        "expand_weight_kind": "importance",
                    },
                ),
                Node(
                    "select",
                    _SelectKernel.ref,
                    base="expand",
                    structural=StructuralDelta.FILTER,
                    mass="free",
                    inputs=(Slice("person", ("age",)),),
                ),
            ),
        )
    )
    registry = KernelRegistry()
    for k in (_FrameKernel(frame), _ExpandKernel(), _SelectKernel()):
        registry.register(k)
    source = tmp_path / "invented.txt"
    source.write_text("invented rows; no external survey")
    store = ContentStore(tmp_path / "lineage-store")
    cold = run_graph(graph, sources={"invented": source}, store=store, kernels=registry)
    replay = run_graph(
        graph,
        sources={"invented": source},
        store=store,
        kernels=registry,
        resume="require",
    )
    assert all(n.hit for n in replay.nodes.values())
    pair_rows = cold.nodes["expand"].receipt["expand"]["household"]
    assert pair_rows == replay.nodes["expand"].receipt["expand"]["household"]
    expanded = cold.population("expand")
    selected = cold.population("select")
    steps = (
        HouseholdExpansion(
            "spi_support_channel",
            tuple(ids),
            tuple(expanded.table("household").household_id),
            tuple(tuple(p) for p in pair_rows),
            ((11, 1), (12, 1), (13, 1)),
        ),
        HouseholdSelection(
            tuple(expanded.table("household").household_id),
            tuple(selected.table("household").household_id),
        ),
    )
    keys = project(
        roots((1, 2, 3)), steps, tuple(selected.table("household").household_id)
    )
    selected_tables = {e: selected.table(e).copy(deep=True) for e in selected.entities}
    selected_tables["household"][IDENTITY_COLUMN] = keys[IDENTITY_COLUMN].array.copy()
    keyed = Frame(
        selected_tables,
        selected.schema,
        {"household": selected.weights_for("household")},
        selected.strata,
        metadata=selected.metadata,
        mass_log=selected.mass_log,
    )
    support_payloads = payloads()
    definition = uk_atomic_assignment_definition(support_payloads, seed=43)
    all_columns = (*columns, Owned("household", IDENTITY_COLUMN, "string"))
    atomic_nodes = atomic_geography_nodes(
        definition, all_columns, base="keyed", emit_validation_artifact=True
    )
    assignment_graph = compile_graph(
        Graph(
            "uk",
            (
                SourceRef("invented", "raw-bytes-v1"),
                *(SourceRef(SOURCES[s], "raw-bytes-v1") for s in support_payloads),
            ),
            (
                Node(
                    "keyed",
                    _FrameKernel.ref,
                    structural=StructuralDelta.CREATE,
                    sources=("invented",),
                    outputs=all_columns,
                ),
                *atomic_nodes,
            ),
        )
    )
    source_paths = {"invented": source}
    for system, payload in support_payloads.items():
        path = tmp_path / (system + ".npz")
        path.write_bytes(payload)
        source_paths[SOURCES[system]] = path
    atomic_registry = KernelRegistry()
    atomic_registry.register(_FrameKernel(keyed))
    register_atomic_geography_kernels(atomic_registry)
    atomic_store = ContentStore(tmp_path / "atomic-store")
    assigned = run_graph(
        assignment_graph,
        sources=source_paths,
        store=atomic_store,
        kernels=atomic_registry,
    )
    required = run_graph(
        assignment_graph,
        sources=source_paths,
        store=atomic_store,
        kernels=atomic_registry,
        resume="require",
    )
    assert len(assigned.nodes) == 7 and all(n.hit for n in required.nodes.values())
    actual = assigned.population("keyed")
    for entity in keyed.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity)[list(keyed.table(entity))],
            keyed.table(entity),
            check_index_type=False,
        )
    np.testing.assert_array_equal(
        actual.weights_for("household").values, keyed.weights_for("household").values
    )
    pd.testing.assert_series_equal(actual.strata, keyed.strata, check_index_type=False)
    assert actual.metadata == keyed.metadata and actual.mass_log == keyed.mass_log
    assert assigned.nodes["geography.gate"].receipt["outcome"] == "pass"
    supports = {
        k: atomic_geography.decode_atomic_support(v)
        for k, v in support_payloads.items()
    }
    reordered = keyed.table("household").iloc[::-1]
    expected = atomic_geography.assign_atomic(reordered, definition, supports)
    for c in expected:
        pd.testing.assert_series_equal(
            expected[c].astype(pd.StringDtype(storage="python")).reset_index(drop=True),
            actual.table("household")[c]
            .iloc[::-1]
            .astype(pd.StringDtype(storage="python"))
            .reset_index(drop=True),
            check_names=False,
        )

    (tmp_path / "acceptance.json").write_text(
        json.dumps(
            {
                "scope": "invented actual shared CREATE/EXPAND/FILTER and atomic graph; pure descriptive lineage helper",
                "lineage_nodes": len(cold.nodes),
                "lineage_required_hits": sum(n.hit for n in replay.nodes.values()),
                "atomic_nodes": len(assigned.nodes),
                "atomic_required_hits": sum(n.hit for n in required.nodes.values()),
                "root_households": frame.n("household"),
                "expanded_households": expanded.n("household"),
                "selected_households": selected.n("household"),
                "source_admission": False,
                "native_inputs": False,
            },
            indent=2,
        )
        + "\n"
    )
